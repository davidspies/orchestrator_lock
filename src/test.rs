use std::time::Duration;
use tokio::time::{sleep, timeout};

use crate::{MutexGuard, OrchestratorMutex, error};

#[tokio::test]
async fn test_basic_functionality() {
    let mut orchestrator = OrchestratorMutex::new(42);
    let (mut granter, mut locker) = orchestrator.add_locker();

    // Grant access and start acquiring in separate task
    let acquire_task = tokio::spawn(async move {
        let mut guard = locker.acquire().await.unwrap();
        assert_eq!(*guard, 42);
        *guard = 84;
        drop(guard);
        locker
    });

    // Give the task time to start
    sleep(Duration::from_millis(10)).await;

    // Grant access
    let grant_future = orchestrator.grant_access(&mut granter).await.unwrap();

    // Wait for locker to release the guard and get it back
    let mut_guard = grant_future.await;
    assert_eq!(*mut_guard, 84);

    // Cleanup
    acquire_task.await.unwrap();
}

#[tokio::test]
async fn test_multiple_lockers() {
    let mut orchestrator = OrchestratorMutex::new(0);
    let (mut granter1, mut locker1) = orchestrator.add_locker();
    let (mut granter2, mut locker2) = orchestrator.add_locker();

    // Set up tasks that increment the counter
    let task1 = tokio::spawn(async move {
        let mut guard = locker1.acquire().await.unwrap();
        *guard += 1;
        sleep(Duration::from_millis(50)).await; // Hold the lock for a bit
        locker1
    });

    let task2 = tokio::spawn(async move {
        let mut guard = locker2.acquire().await.unwrap();
        *guard += 10;
        locker2
    });

    // Grant access to first locker
    let grant_future1 = orchestrator.grant_access(&mut granter1).await.unwrap();

    // Wait for first locker to finish and get the lock back
    let mut_guard = grant_future1.await;
    assert_eq!(*mut_guard, 1);
    drop(mut_guard);

    // Now grant access to second locker
    let grant_future2 = orchestrator.grant_access(&mut granter2).await.unwrap();

    // Wait for second locker to finish and get the lock back
    let mut_guard = grant_future2.await;
    assert_eq!(*mut_guard, 11); // 1 + 10
    drop(mut_guard);

    // Cleanup
    task1.await.unwrap();
    task2.await.unwrap();
}

#[tokio::test]
async fn test_try_acquire() {
    let mut orchestrator = OrchestratorMutex::new(42);
    let (mut granter, mut locker) = orchestrator.add_locker();

    // Initially try_acquire should fail with AccessDenied
    match locker.try_acquire() {
        Err(error::TryAcquireError::AccessDenied) => {}
        _ => panic!("Expected AccessDenied"),
    }

    // Grant access in a separate task
    let grant_task = tokio::spawn(async move {
        let grant_future = orchestrator.grant_access(&mut granter).await.unwrap();
        let mut_guard = grant_future.await;
        assert_eq!(*mut_guard, 84); // Should be modified by the locker
    });

    // Wait for access to be granted
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Now try_acquire should succeed
    let mut guard = locker.try_acquire().unwrap();
    assert_eq!(*guard, 42);
    *guard = 84; // Modify the value
    drop(guard);

    // Wait for grant task to complete
    grant_task.await.unwrap();
}

#[tokio::test]
async fn test_direct_acquisition() {
    let mut orchestrator = OrchestratorMutex::new(42);

    // Direct acquire
    let mut guard = orchestrator.acquire().await;
    assert_eq!(*guard, 42);
    *guard = 84;
    drop(guard);

    // Direct try_acquire
    let mut guard = orchestrator.try_acquire().expect("try_acquire failed");
    assert_eq!(*guard, 84);
    *guard = 126;
    drop(guard);

    // Verify value again
    let guard = orchestrator.acquire().await;
    assert_eq!(*guard, 126);
}

#[tokio::test]
async fn test_orchestrator_dropped() {
    let orchestrator = OrchestratorMutex::new(42);
    let (_granter, mut locker) = orchestrator.add_locker();

    // Drop the orchestrator
    drop(orchestrator);

    // Locker should now fail to acquire
    let result = timeout(Duration::from_millis(100), locker.acquire()).await;
    assert!(result.is_ok(), "acquire didn't complete");
    assert!(result.unwrap().is_none(), "acquire didn't return None");

    // try_acquire should also fail
    match locker.try_acquire() {
        Err(error::TryAcquireError::Inaccessible) => {}
        _ => panic!("Expected Inaccessible error"),
    }
}

#[tokio::test]
async fn test_granter_dropped() {
    let orchestrator = OrchestratorMutex::new(42);
    let (granter, mut locker) = orchestrator.add_locker();

    // Drop the granter
    drop(granter);

    // Locker should now fail to acquire
    let result = timeout(Duration::from_millis(100), locker.acquire()).await;
    assert!(result.is_ok(), "acquire didn't complete");
    assert!(result.unwrap().is_none(), "acquire didn't return None");

    // try_acquire should also fail
    match locker.try_acquire() {
        Err(error::TryAcquireError::Inaccessible) => {}
        _ => panic!("Expected Inaccessible error"),
    }
}

#[tokio::test]
async fn test_mutex_guard_into_owned() {
    let mut orchestrator = OrchestratorMutex::new(42);
    let (mut granter, mut locker) = orchestrator.add_locker();

    let grant_task = tokio::spawn(async move {
        let grant_future = orchestrator.grant_access(&mut granter).await.unwrap();
        sleep(Duration::from_millis(50)).await; // Give time for locker to process
        let mut_guard = grant_future.await;
        assert_eq!(*mut_guard, 84);
    });

    // Acquire the guard, convert to owned, and modify
    let guard = locker.acquire().await.unwrap();
    let mut owned_guard = MutexGuard::into_owned_guard(guard);
    assert_eq!(*owned_guard, 42);
    *owned_guard = 84;
    drop(owned_guard);

    // Wait for orchestrator task to finish
    grant_task.await.unwrap();
}

#[tokio::test]
async fn test_grant_access_with_dropped_locker() {
    let mut orchestrator = OrchestratorMutex::new(42);
    let (mut granter, locker) = orchestrator.add_locker();

    // Drop the locker
    drop(locker);

    // Attempt to grant access should fail
    let result = orchestrator.grant_access(&mut granter).await;
    assert!(
        result.is_err(),
        "grant_access should return Err when MutexLocker is dropped"
    );
    drop(result);

    // We should still be able to directly acquire the lock
    let guard = orchestrator.acquire().await;
    assert_eq!(*guard, 42);
}

#[tokio::test]
async fn test_concurrent_access_and_modifications() {
    const NUM_LOCKERS: usize = 5;
    const ITERATIONS_PER_LOCKER: usize = 10;

    let mut orchestrator = OrchestratorMutex::new(0);
    let mut granters = Vec::with_capacity(NUM_LOCKERS);
    let mut locker_handles = Vec::with_capacity(NUM_LOCKERS);

    // Create lockers and spawn tasks
    for i in 0..NUM_LOCKERS {
        let (granter, mut locker) = orchestrator.add_locker();
        granters.push(granter);

        let task = tokio::spawn(async move {
            let mut expected_value = i as i32; // Start value depends on locker index

            for _ in 0..ITERATIONS_PER_LOCKER {
                if let Some(mut guard) = locker.acquire().await {
                    // Verify current value matches our expectation
                    assert_eq!(
                        *guard, expected_value,
                        "Locker {} expected value {} but got {}",
                        i, expected_value, *guard
                    );

                    // Increment the counter
                    *guard += 1;

                    // Next time we expect the value to have been incremented by each locker
                    expected_value += NUM_LOCKERS as i32;

                    // Simulate some work
                    sleep(Duration::from_millis(5)).await;
                } else {
                    panic!("Locker {} failed to acquire lock", i);
                }
            }
            locker
        });

        locker_handles.push(task);
    }

    // Grant access to lockers in a round-robin fashion
    for iteration in 0..ITERATIONS_PER_LOCKER {
        for i in 0..NUM_LOCKERS {
            let grant_future = orchestrator.grant_access(&mut granters[i]).await.unwrap();
            let guard = grant_future.await;

            // After each complete round, value should have increased by NUM_LOCKERS
            if i == NUM_LOCKERS - 1 && iteration < ITERATIONS_PER_LOCKER - 1 {
                assert_eq!(
                    *guard,
                    ((iteration + 1) * NUM_LOCKERS) as i32,
                    "After round {}, expected value {} but got {}",
                    iteration + 1,
                    ((iteration + 1) * NUM_LOCKERS),
                    *guard
                );
            }
        }
    }

    // Wait for all lockers to finish
    for handle in locker_handles {
        let _ = handle.await.unwrap();
    }

    // Check final value
    let guard = orchestrator.acquire().await;
    assert_eq!(*guard, (NUM_LOCKERS * ITERATIONS_PER_LOCKER) as i32);
}
