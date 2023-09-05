import threading

# Define a global event to signal the thread when the variable is ready
variable_ready_event = threading.Event()

# The variable that will be populated in the main process
shared_variable = None

# Function that represents your thread
def my_thread_function():
    global shared_variable

    # Wait for the event to be set, indicating that the variable is ready
    print(f"variable_ready_event is set: {variable_ready_event.is_set()}")
    variable_ready_event.wait()
    print(f"variable_ready_event is set 2: {variable_ready_event.is_set()}")

    # Now you can safely use the updated variable
    print(f"Thread: The shared variable is {shared_variable}")

# Create and start the thread
thread = threading.Thread(target=my_thread_function)
thread.start()

# Simulate populating the variable in the main process
shared_variable = "Hello, World!"

# Set the event to signal the thread that the variable is ready
variable_ready_event.set()

# Wait for the thread to finish
thread.join()

print("Main process: Thread has finished")