# Gravity-Simulations
A couple gravity simulations

N-body Sim
- Simulates randomly generated bodies' gravitational affect on eachother
- Very quickly put together and not at all optimized.
- Uses the Barnes-Hut algorithm to reduce calculations and velocity verlet for more accuracy.
- Preset generation values, changeable in code
- Controls: 
  - WASD for camera movement
  - ESC to close

Ring Sim
- Simulates moons' gravitational affect on ring particles
- Uses velocity verlet for accurate/stable orbits
- Time warp and body creation
- Random generation and resets
- Trajectory drawing of moons and created bodies
- 'Collisions' between particles and planet/moons/created bodies
- Controls:
  - WASD for camera movement
  - ESC to close
  - L/J to increase/decrease time warp, K to reset time warp
  - F to randomize, R to reset current system
  - Left click hold to create body
      - Up/down arrows to increase/decrease mass
      - Right/left arrows to increase/decrease size
      - Shift to increase increments for mass and velocity
      - Release left click to finalize body

Saturn Ring Sim
- uses GPU acceleration with Taichi
- Wisdom Holman integration for extreme orbital stability (using elliptical lagrange coefficients to propagate kepler orbits)
- no time warp, body creation, or camera controls
- particle-particle statistical collisions (uses an assumed global optical depth- better for simulating individual rings with low tau variance)
