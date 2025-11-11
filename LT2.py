import numpy as np
from scipy.optimize import brentq
from scipy import interpolate
from dataclasses import dataclass
import time
import matplotlib.pyplot as plt

# Define the material coordinate range
r_range = np.linspace(1.0, 2.0, 32)  # 64 points from 1 to 2 with step 1/63

def plot_and_save(states, r_range, time, n=20, filename="cylinder_LT2.png"):
    """
    Plots various simulation results for n evenly spaced states and saves the figure.
    """
    print(f"\nGenerating and saving plot to {filename}...")
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("LT2 (hoop driven hoop stress)", fontsize=16)
    
    num_total_states = len(states)
    if n >= num_total_states:
        plot_indices = np.arange(num_total_states)
    else:
        # Ensure the first and last states are included
        plot_indices = np.unique(np.linspace(0, num_total_states - 1, n, dtype=int))

    states_to_plot = [(i, states[i]) for i in plot_indices]
    colors = plt.cm.viridis(np.linspace(0, 1, len(states_to_plot)))

    # 1. Plot Growth Factor (gt) vs. R
    ax = axs[0, 0]
    for i, (state_idx, state) in enumerate(states_to_plot):
        ax.plot(r_range, state.gt, color=colors[i], label=f"t= {time[state_idx]:.3f}")
    ax.set_title("Cumulative Growth (gt) vs. Material Coordinate (R)")
    ax.set_xlabel("R")
    ax.set_ylabel(r"$g_\theta(R)$")
    ax.grid(True)

    # 2. Plot Displacement (r) vs. R
    ax = axs[0, 1]
    for i, (state_idx, state) in enumerate(states_to_plot):
        ri = state.find_inner_radius()
        r_values = [state.compute_r(ri, s) for s in r_range]
        ax.plot(r_range, r_values, color=colors[i], label=f"Iter {state_idx}")
    ax.set_title("Displacement (r) vs. Material Coordinate (R)")
    ax.set_xlabel("R")
    ax.set_ylabel("r(R)")
    ax.grid(True)

    # 3. Plot Radial Stress vs. R for selected states
    ax = axs[1, 0]
    stress_plot_points = np.linspace(1.0, 2.0, 50)
    
    for i, (state_idx, state) in enumerate(states_to_plot):
        ri = state.find_inner_radius()
        stress_values = [state.radial_stress(ri, s) * (s / state.compute_r(ri, s)) for s in stress_plot_points]
        ax.plot(stress_plot_points, stress_values, color=colors[i], label=f"Iter {state_idx}")
    
    ax.set_title("Radial Stress (Cauchy) vs. Material Coordinate (R)")
    ax.set_xlabel("R")
    ax.set_ylabel("Radial Stress")
    ax.grid(True)

    # 4. Plot Growth Trigger vs. R
    ax = axs[1, 1]
    for i, (state_idx, state) in enumerate(states_to_plot):
        ri = state.find_inner_radius()
        triggers = [state.compute_dgt(ri, s) for s in r_range]
        ax.plot(r_range, triggers, color=colors[i], label=f"Iter {state_idx}")
    ax.axhline(1, color='r', linestyle='--', label="Equilibrium (0)")
    ax.set_title("Growth Trigger vs. Material Coordinate (R)")
    ax.set_xlabel("R")
    ax.set_ylabel(r"dgt = $\frac{R}{r}g_\theta - a_r^*$")
    ax.grid(True)

    # 5. Plot Circumferential Stress vs. R for selected states
    ax = axs[2, 0]
    for i, (state_idx, state) in enumerate(states_to_plot):
        ri = state.find_inner_radius()
        stress_values = [state.angular_stress(ri, s) * (state.compute_r(ri, s) / s) / (state.gr_interp(s) * state.gt_interp(s)) for s in stress_plot_points]
        ax.plot(stress_plot_points, stress_values, color=colors[i], label=f"Iter {state_idx}")
    ax.set_title("Circumferential Stress (Cauchy) vs. Material Coordinate (R)")
    ax.set_xlabel("R")
    ax.set_ylabel("Circumferential Stress")
    ax.grid(True)

    # 6. Plot p
    ax = axs[2, 1]
    for i, (state_idx, state) in enumerate(states_to_plot):
        ri = state.find_inner_radius()
        p_values = [state.compute_p(ri, s) for s in r_range]
        ax.plot(r_range, p_values, color=colors[i], label=f"Iter {state_idx}")
    ax.set_title("Pressure (p) vs. Material Coordinate (R)")
    ax.set_xlabel("R")
    ax.set_ylabel("p(R)")
    ax.grid(True)

    # Add a single legend for the iterations
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', title="Time")
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(filename)
    plt.close(fig)
    print("✓ Plot saved.")

class FastState:
    """Optimized state with precomputed values"""
    def __init__(self, _Ri, gr, gt, bc, mu, gMax, lambdaCrit, gamma, tau):
        self._Ri = _Ri                          # Inner radius
        self.gr = np.array(gr)                  # Radial growth factor
        self.gt = np.array(gt)                  # Circumferential growth factor
        self.bc = bc                            # Boundary condition
        self.mu = mu                            # Shear modulus     
        self.gMax = gMax                        # Growth factor setpoint
        self.lambdaCrit = lambdaCrit            # Strain setpoint
        self.gamma = gamma                      # Growth exponent
        self.tau = tau                          # Growth timescale

        #Interpolate gr
        self.gr_interp = interpolate.interp1d(
            r_range, self.gr, kind='linear',
            bounds_error=False, fill_value='extrapolate'
        )

        # Interpolate gt
        self.gt_interp = interpolate.interp1d(
            r_range, self.gt, kind='linear',
            bounds_error=False, fill_value='extrapolate'
        )
   
    def compute_r(self, ri, s):
        """Solve r = sqrt(ri^2 + 2*∫s^2*gr*gt^2ds)"""
        if s <= self._Ri:
            return ri
        
        # Simple trapezoidal integration
        x = np.linspace(self._Ri, s, 200)
        integrand = self.gr_interp(x) * self.gt_interp(x) * x
        integral = np.trapezoid(integrand, x)
        return np.sqrt(ri**2 + 2 * integral)
    
    def compute_p(self, ri, s):
        """ Solve p
        """

        # Boundary condition for pressure at the inner radius Ri
        gr_Ri = self.gr_interp(self._Ri)
        gt_Ri = self.gt_interp(self._Ri)
        
        # From radial stress boundary condition: sigma_rr(Ri) = bc
        # sigma_rr = (mu/gr)*(s/r)^2 * gt - p
        # bc = (mu/gr_Ri)*(Ri/ri)^2 * gt_Ri - p_i => p_i = (mu/gr_Ri)*(Ri/ri)^2 * gt_Ri - bc
        p_i = self.bc * gr_Ri * gt_Ri - self.mu * (self._Ri / ri)**2 * gt_Ri**2

        if s <= self._Ri:
            return p_i

        # Define integration points
        x = np.linspace(self._Ri, s, 200)
        
        # Pre-calculate values needed for the integral
        r_vals = np.array([self.compute_r(ri, si) for si in x])
        gr_vals = self.gr_interp(x)
        gt_vals = self.gt_interp(x)
        dgr_ds = np.gradient(gr_vals, x)
        dgt_ds = np.gradient(gt_vals, x)
        # dgr_ds, dgt_ds = self._get_derivatives(s)

        # Calculate each term of the integrand
        term1 = 2 * (x / r_vals**2) * (gt_vals / gr_vals)
        term2 = - (x**3 / r_vals**4) * gt_vals**2
        term3 = (x**2 / r_vals**2) / gr_vals * dgt_ds
        term4 = -(x**2 / r_vals**2) * (gt_vals / gr_vals**2) * dgr_ds
        term5 = - 1 / (x * gt_vals**2)
        
        integrand = self.mu * (term1 + term2 + term3 + term4 + term5)
        
        integral = np.trapezoid(integrand, x)

        # Calculate the constant C = p(Ri) / (gr(Ri) * gt(Ri)^2)
        C = p_i / (gr_Ri * gt_Ri**2)
        
        # Final pressure calculation
        gr_s = self.gr_interp(s)
        gt_s = self.gt_interp(s)
        
        return gr_s * gt_s**2 * (C - integral)
    
    def radial_stress(self, ri, s):
        """Compute radial stress: Pʳᴿ = μ(R²/r²)(gₒ²/gᵣ) + p(r²/(R²gᵣgₒ²))"""
        r_val = self.compute_r(ri, s)
        gr_val = self.gr_interp(s)
        gt_val = self.gt_interp(s)
        p_val = self.compute_p(ri, s)
        
        term1 = self.mu * (s / r_val) * (gt_val / gr_val)
        term2 = p_val * (r_val / s) / (gr_val * gt_val)
        
        return term1 + term2
    
    def angular_stress(self, ri, s):
        r_val = self.compute_r(ri, s)
        gt_val = self.gt_interp(s)
        p_val = self.compute_p(ri, s)
        
        term1 = self.mu * (r_val / s) / (gt_val**2)
        term2 = p_val * (s / r_val)
        
        return term1 + term2

    def find_inner_radius(self):
        """Find inner radius using root finding"""
        def objective(ri):
            return self.radial_stress(ri, 2.0)
        
        # Use a good initial guess based on the previous value
        try:
            return brentq(objective, self._Ri - 0.5, self._Ri + 0.5, 
                         xtol=1e-6, maxiter=20)
        except:
            # If that fails, try a wider bracket
            return brentq(objective, 0.5, 2.5, xtol=1e-6)
    
    def compute_dgt(self, ri, s):
        """Compute growth rate based on circumferential stress."""
        r_val = self.compute_r(ri, s)
        gr_val = self.gr_interp(s)
        gt_val = self.gt_interp(s)
        p_val = self.compute_p(ri, s)

        # This simplifies to dgt = (g_theta / tau) * (1/sigma_star) * (stress_term - sigma_star)
        
        stress_term = ((self.mu * (r_val / s)**2 / gt_val**2 + p_val)/ (gr_val * gt_val))

        dgt = self.tau * (1 / self.lambdaCrit) * (stress_term - self.lambdaCrit) + 1

        return dgt
    
    def update(self):
        """Create updated state"""
        ri = self.find_inner_radius()

        # Vectorized dgt computation
        dgt = np.array([self.compute_dgt(ri, s) for s in r_range])
        new_gt = self.gt * dgt

        return FastState(
            self._Ri, self.gr, new_gt, self.bc, self.mu,
            self.gMax, self.lambdaCrit, self.gamma, self.tau
        )
    
    def __str__(self):
        return (f"State(_Ri={self._Ri}, gr={self.gr}, "
                f"gf=[{self.gt[0]:.6f}, {self.gt[-1]:.6f}], "
                f"bc={self.bc}, mu={self.mu})")

def main():
    # Initialize base state
    initial_gf = np.ones_like(r_range)
    initial_gr = np.ones_like(r_range)  # No initial growth

    base_state = FastState(
        _Ri=1.0,
        gr=initial_gr,
        gt=initial_gf,
        bc=-0.05,
        mu=1.0,
        gMax=1.5,
        lambdaCrit=0.5,
        gamma=1,
        tau=0.025
    )
    
    print("Initial state:", base_state)
    
    # Initial calculations
    ri = base_state.find_inner_radius()
    stress_points = np.arange(1, 2.1, 0.1)
    stress_data = [base_state.radial_stress(ri, x) for x in stress_points]
    
    print(f"\nInitial inner radius: {ri:.15f}")
    print("Initial stress data:")
    print("[" + ",".join(f"{x:.15f}" for x in stress_data) + "]")
    
    # Iterate state updates
    print("\nIterating states...")
    states = [base_state]
    current_state = base_state
    
    n = 10
    for i in range(n):
        print(f"  Iteration {i+1}/{n}", end="", flush=True)
        start = time.time()
        current_state = current_state.update()
        states.append(current_state)
        print(f" - {time.time()-start:.3f}s")
    
    # Final calculations
    print("\nFinal state:", states[-1])
    
    last_state = states[-1]
    ri = last_state.find_inner_radius()
    stress_data = [last_state.radial_stress(ri, x) for x in stress_points]
    
    # Convert PK1 stress to Cauchy stress
    # Multiply by (r/R)^2
    stress_data = [
        stress_data[i] * (last_state.compute_r(ri, stress_points[i]) / stress_points[i])**2
        for i in range(len(stress_points))
    ]    

    print("\nFinal stress data:")
    print("[" + ",".join(f"{x:.15f}" for x in stress_data) + "]")
    
    # Final gs values
    gs = [state.gr_interp(np.arange(1, 2 + 1/64, 1/64)).tolist() 
          for state in states]
    print("\nFirst gs:", gs[0][:3], "...")
    print("Last gs:", gs[-1][:3], "...")
    
    time_points = np.arange(n) * base_state.tau
    plot_and_save(states[:-1], r_range, time=time_points)

if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"\n✓ Total execution time: {total_time:.2f} seconds")
