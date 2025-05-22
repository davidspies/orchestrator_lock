//! `orchestrator_lock` provides a specialized mutex implementation for scenarios
//! where fine-grained control over mutex access is required. Unlike a standard
//! mutex where any code with a reference can attempt to acquire the lock, this
//! implementation separates the concerns of lock orchestration from lock usage.
//!
//! # Core Concepts
//!
//! * **OrchestratorMutex**: The central coordinator that owns the protected value and
//!   controls access to it.
//!
//! * **Granter**: A capability token that allows the orchestrator to grant lock access
//!   to a specific locker.
//!
//! * **MutexLocker**: The component that can acquire and use the lock, but only when
//!   explicitly granted access by the orchestrator.
//!
//! * **MutexGuard**: Provides access to the protected value, similar to a standard
//!   mutex guard.
//!
//! # Example
//! ```
//! use tokio::time::Duration;
//!
//! use orchestrator_lock::OrchestratorMutex;
//!
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() {
//!     // Create a shared counter with initial value 0
//!     let mut orchestrator = OrchestratorMutex::new(0);
//!     
//!     // Create two granter/locker pairs
//!     let (mut granter1, mut locker1) = orchestrator.add_locker();
//!     let (mut granter2, mut locker2) = orchestrator.add_locker();
//!     
//!     // Task 1: Increments by 1 each time
//!     let task1 = tokio::spawn(async move {
//!         let expected = [0, 2, 6];
//!         for i in 0..3 {
//!             if let Some(mut guard) = locker1.acquire().await {
//!                 assert_eq!(*guard, expected[i]);
//!                 *guard += 1;
//!                 tokio::time::sleep(Duration::from_millis(10)).await;
//!             }
//!         }
//!         locker1
//!     });
//!     
//!     // Task 2: Multiplies by 2 each time
//!     let task2 = tokio::spawn(async move {
//!         let expected = [1, 3, 7];
//!         for i in 0..3 {
//!             if let Some(mut guard) = locker2.acquire().await {
//!                 assert_eq!(*guard, expected[i]);
//!                 *guard *= 2;
//!                 tokio::time::sleep(Duration::from_millis(10)).await;
//!             }
//!         }
//!         locker2
//!     });
//!     
//!     // Orchestration: Alternate between the two tasks
//!     for i in 0..3 {
//!         // Grant access to task 1
//!         let task1_holding = orchestrator.grant_access(&mut granter1).await.unwrap();
//!         task1_holding.await;
//!         
//!         // Grant access to task 2
//!         let task2_holding = orchestrator.grant_access(&mut granter2).await.unwrap();
//!         task2_holding.await;
//!     }
//!     assert_eq!(*orchestrator.acquire().await, 14);
//!     // Clean up
//!     let _ = task1.await.unwrap();
//!     let _ = task2.await.unwrap();
//! }
//! ```

use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Weak};

use awaitable_bool::AwaitableBool;
use tokio::select;
use tokio::sync::{Mutex, Notify};

pub mod error {
    #[derive(Debug, PartialEq, Eq)]
    pub struct GrantError;

    #[derive(Debug, PartialEq, Eq)]
    pub enum TryAcquireError {
        /// [grant_access](super::OrchestratorMutex::grant_access) has not been
        /// called with the corresponding [Granter](super::Granter).
        AccessDenied,
        /// Either the corresponding [Granter](super::Granter) or the
        /// [OrchestratorMutex](super::OrchestratorMutex) has been dropped.
        Inaccessible,
    }
}

pub struct OrchestratorMutex<T> {
    inner: Arc<Mutex<T>>,
    dropped: Arc<AwaitableBool>,
}

impl<T> Drop for OrchestratorMutex<T> {
    fn drop(&mut self) {
        self.dropped.set_true();
    }
}

pub struct OwnedMutexGuard<T> {
    // Field ordering ensures that the inner guard is dropped before the
    // finished notification is sent.
    inner: tokio::sync::OwnedMutexGuard<T>,
    _finished: Finished,
}

struct Finished(Arc<Notify>);

pub struct Granter<T> {
    inner: Weak<Mutex<T>>,
    tx: relay_channel::Sender<OwnedMutexGuard<T>>,
}

pub struct MutexLocker<T> {
    mutex_dropped: Arc<AwaitableBool>,
    rx: relay_channel::Receiver<OwnedMutexGuard<T>>,
}

pub struct MutexGuard<'a, T> {
    guard: OwnedMutexGuard<T>,
    _locker: &'a mut MutexLocker<T>,
}

impl<T> OrchestratorMutex<T> {
    pub fn new(value: T) -> Self {
        let dropped = Arc::new(AwaitableBool::new(false));
        Self {
            inner: Arc::new(Mutex::new(value)),
            dropped,
        }
    }

    pub fn add_locker(&self) -> (Granter<T>, MutexLocker<T>) {
        let (tx, rx) = relay_channel::channel();
        let inner = Arc::downgrade(&self.inner);
        let mutex_dropped = self.dropped.clone();
        (Granter { inner, tx }, MutexLocker { mutex_dropped, rx })
    }

    /// Directly acquire the underlying lock.
    pub async fn acquire(&self) -> tokio::sync::MutexGuard<'_, T> {
        self.inner.lock().await
    }

    pub fn blocking_acquire(&self) -> tokio::sync::MutexGuard<'_, T> {
        self.inner.blocking_lock()
    }

    /// Attempt to acquire the underlying lock, failing if the lock is already
    /// held.
    pub fn try_acquire(&self) -> Result<tokio::sync::MutexGuard<'_, T>, tokio::sync::TryLockError> {
        self.inner.try_lock()
    }

    /// Grants lock access to the [MutexLocker] corresponding to the provided
    /// [Granter].
    ///
    /// This function returns [Ok] once the corresponding [MutexLocker] has
    /// called
    /// [acquire](MutexLocker::acquire) (or [Err] if the [MutexLocker] has been
    /// dropped).
    /// The [Ok] variant contains a future which waits for the acquiring task to
    /// drop its [MutexGuard].
    ///
    /// If the future in the [Ok] variant is dropped, the next call to
    /// [grant_access](Self::grant_access) will have to wait for the current
    /// [MutexGuard] to be dropped before it can grant access to the next
    /// [MutexLocker]. If this is called multiple times in parallel, the
    /// order in which the [MutexLocker]s are granted access is unspecified.
    ///
    /// # Panics
    /// Panics if `granter` was created from a different [OrchestratorMutex].
    pub async fn grant_access(
        &self,
        granter: &mut Granter<T>,
    ) -> Result<impl Future<Output = ()>, error::GrantError> {
        assert!(
            Weak::ptr_eq(&granter.inner, &Arc::downgrade(&self.inner)),
            "Granter is not associated with this OrchestratorMutex"
        );
        let inner_guard = self.inner.clone().lock_owned().await;
        let finished = Arc::new(Notify::new());
        let guard = OwnedMutexGuard {
            inner: inner_guard,
            _finished: Finished(Arc::clone(&finished)),
        };
        match granter.tx.send(guard).await {
            Ok(()) => Ok(async move { finished.notified().await }),
            Err(relay_channel::error::SendError(_)) => Err(error::GrantError),
        }
    }
}

impl<T> MutexLocker<T> {
    /// Returns [None] if either the corresponding [Granter] or the
    /// [OrchestratorMutex] has been dropped.
    pub async fn acquire(&mut self) -> Option<MutexGuard<'_, T>> {
        let result = select! {
            result = self.rx.recv() => result,
            () = self.mutex_dropped.wait_true() => None,
        };
        Some(MutexGuard {
            guard: result?,
            _locker: self,
        })
    }

    pub fn try_acquire(&mut self) -> Result<MutexGuard<'_, T>, error::TryAcquireError> {
        match self.rx.try_recv() {
            Ok(guard) => Ok(MutexGuard {
                guard,
                _locker: self,
            }),
            Err(relay_channel::error::TryRecvError::Empty) => {
                if self.mutex_dropped.is_true() {
                    Err(error::TryAcquireError::Inaccessible)
                } else {
                    Err(error::TryAcquireError::AccessDenied)
                }
            }
            Err(relay_channel::error::TryRecvError::Disconnected) => {
                Err(error::TryAcquireError::Inaccessible)
            }
        }
    }
}

impl<T> Deref for OwnedMutexGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> DerefMut for OwnedMutexGuard<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl Drop for Finished {
    fn drop(&mut self) {
        self.0.notify_one();
    }
}

impl<T> MutexGuard<'_, T> {
    /// The lifetime parameter on [MutexGuard] is only for convenience (to help
    /// avoid having multiple parallel calls to [acquire](MutexLocker::acquire)
    /// and [try_acquire](MutexLocker::try_acquire)). The caller can choose to
    /// instead
    /// use this function to unwrap the underlying [OwnedMutexGuard] if it's
    /// more convenient not to deal with the lifetime.
    pub fn into_owned_guard(this: Self) -> OwnedMutexGuard<T> {
        this.guard
    }
}

impl<T> Deref for MutexGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl<T> DerefMut for MutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}

#[cfg(test)]
mod test;
