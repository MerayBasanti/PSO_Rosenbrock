
import numpy as np

# Define the Rosenbrock function (the function to be minimized)
def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

# Define the PSO algorithm
def pso(func, n_particles, n_dimensions, n_iterations, bounds):
    # Initialize the particles with random positions and velocities within the specified bounds
    particles = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_particles, n_dimensions))
    velocities = np.zeros((n_particles, n_dimensions))

    # Initialize the personal best positions and values
    personal_best_positions = particles.copy()
    personal_best_values = np.array([func(p) for p in personal_best_positions])

    # Initialize the global best position and value
    global_best_index = np.argmin(personal_best_values)
    global_best_position = personal_best_positions[global_best_index]
    global_best_value = personal_best_values[global_best_index]

    # PSO parameters
    w = 0.5  # Inertia weight
    c1 = 2.5
    c2 = 1.5

    for _ in range(n_iterations):
        for i in range(n_particles):
            # Update particle velocities
            r1, r2 = np.random.rand(2)
            cognitive_velocity = c1 * r1 * (personal_best_positions[i] - particles[i])
            social_velocity = c2 * r2 * (global_best_position - particles[i])
            velocities[i] = w * velocities[i] + cognitive_velocity + social_velocity

            # Update particle positions
            particles[i] += velocities[i]

            # Ensure particles stay within bounds
            particles[i] = np.clip(particles[i], bounds[:, 0], bounds[:, 1])

            # Evaluate the new position
            current_value = func(particles[i])

            # Update personal best
            if current_value < personal_best_values[i]:
                personal_best_values[i] = current_value
                personal_best_positions[i] = particles[i]

                # Update global best
                if current_value < global_best_value:
                    global_best_value = current_value
                    global_best_position = particles[i]

    return global_best_position, global_best_value

if __name__ == "__main__":
    n_particles = 30
    n_dimensions = 2
    n_iterations = 100
    bounds = np.array([[-5.12, 5.12], [-5.12, 5.12]])  # Search space bounds

    best_position, best_value = pso(rosenbrock, n_particles, n_dimensions, n_iterations, bounds)

    print("Best solution found:")
    print("Position:", best_position)
    print("Value:", best_value)



