import jax.numpy as jnp

def Rz(angle):
    """Create a 3x3 rotation matrix about the z-axis."""
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    return jnp.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

class ThrdOrderRefFilter:
    """Third-order reference filter for guidance using JAX.
    
    Attributes
    ----------
    eta_d : jnp.ndarray
        3D-array of desired vessel pose in NED-frame.
    eta_d_dot : jnp.ndarray
        3D-array of desired vessel velocity in NED-frame.
    eta_d_ddot : jnp.ndarray
        3D-array of desired vessel acceleration in NED-frame.
    """

    def __init__(self, dt, omega=[0.2, 0.2, 0.2], initial_eta=None):
        self._dt = dt
        self.eta_d = jnp.zeros(3) if initial_eta is None else jnp.array(initial_eta)
        self.eta_d_dot = jnp.zeros(3)
        self.eta_d_ddot = jnp.zeros(3)
        self._eta_r = self.eta_d.copy()
        self._x = jnp.concatenate([self.eta_d, self.eta_d_dot, self.eta_d_ddot])
        self._delta = jnp.eye(3)
        self._w = jnp.diag(jnp.array(omega))
        O3x3 = jnp.zeros((3, 3))
        self.Ad = jnp.block([
            [O3x3,       jnp.eye(3), O3x3],
            [O3x3,       O3x3,       jnp.eye(3)],
            [-self._w**3, -(2*self._delta + jnp.eye(3)) @ (self._w**2), -(2*self._delta + jnp.eye(3)) @ self._w]
        ])
        self.Bd = jnp.block([
            [O3x3],
            [O3x3],
            [self._w**3]
        ])

    def get_eta_d(self):
        """Get desired pose in NED-frame."""
        return self.eta_d

    def get_eta_d_dot(self):
        """Get desired velocity in NED-frame."""
        return self.eta_d_dot

    def get_eta_d_ddot(self):
        """Get desired acceleration in NED-frame."""
        return self.eta_d_ddot

    def get_nu_d(self):
        """Get desired velocity in body-frame."""
        psi = self.eta_d[-1]
        return Rz(psi).T @ self.eta_d_dot

    def set_eta_r(self, eta_r):
        """Set the reference pose.

        Parameters
        ----------
        eta_r : array_like
            Reference vessel pose in surge, sway and yaw.
        """
        self._eta_r = jnp.array(eta_r)

    def update(self):
        """Update the desired position."""
        x_dot = self.Ad @ self._x + self.Bd @ self._eta_r
        self._x = self._x + self._dt * x_dot
        self.eta_d = self._x[:3]
        self.eta_d_dot = self._x[3:6]
        self.eta_d_ddot = self._x[6:]
