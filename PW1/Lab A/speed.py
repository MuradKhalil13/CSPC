def compare_simulation_speed():
    loop_start = time.perf_counter()
    loop_result = simulate_loop(1000, 0.4)
    loop_time = time.perf_counter() - loop_start

    numpy_start = time.perf_counter()
    numpy_result = simulate(1000, 0.4)
    numpy_time = time.perf_counter() - numpy_start

    improvement = loop_time / numpy_time

    print(f"Pure Python time: {loop_time:.6f} seconds")
    print(f"NumPy time: {numpy_time:.6f} seconds")
    print(f"NumPy is approximately {improvement:.0f}x faster.")
    
compare_simulation_speed()