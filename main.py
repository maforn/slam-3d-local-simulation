import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import os

class SLAMSimulator3D:
    def __init__(self, n_poses=100, radius=10, height_variation=2):
        """
        Initialize the 3D SLAM simulator
        
        Args:
            n_poses: number of poses of the path
            radius: radius of the circular path
            height_variation: variation on the vertical axis of the path
        """
        self.n_poses = n_poses
        self.radius = radius
        self.height_variation = height_variation
        
        # Noise parameters
        self.odometry_noise = 0.3
        self.sensor_noise = 0.1
        
        # Loop closure threshold for the final recognition
        self.loop_closure_threshold = 2.0
        
        # Generate ground truth, odometry and landmarks
        self.generate_trajectory()
        
    def generate_trajectory(self):
        """Generate ground truth, odometry and landmarks"""
        
        # Ground ground truth: perfect 3D circular path
        t = np.linspace(0, 2*np.pi, self.n_poses)
        self.ground_truth = np.zeros((self.n_poses, 3))
        self.ground_truth[:, 0] = self.radius * np.cos(t)
        self.ground_truth[:, 1] = self.radius * np.sin(t)
        self.ground_truth[:, 2] = self.height_variation * np.sin(2*t)
        
        # Odometry with noise and systematic drift
        self.odometry = np.zeros((self.n_poses, 3))
        self.odometry[0] = self.ground_truth[0].copy()
        
        # Add systematic bias for realistic drift
        drift_bias = np.array([0.02, 0.015, 0.005])
        
        for i in range(1, self.n_poses):
            true_motion = self.ground_truth[i] - self.ground_truth[i-1]
            
            # Add noise + systematic drift
            noise = np.random.randn(3) * self.odometry_noise
            noisy_motion = true_motion + noise + drift_bias
            
            # Cumulate the drift
            self.odometry[i] = self.odometry[i-1] + noisy_motion
        
        # Generate landmark observations
        self.generate_observations()
        
    def generate_observations(self):
        """Generate obseravations of the landmarks"""
        # Create fixed landmarks in 3D space
        n_landmarks = 20
        angles = np.linspace(0, 2*np.pi, n_landmarks)
        self.landmarks = np.zeros((n_landmarks, 3))
        
        # Place landmark alongside the path
        for i, angle in enumerate(angles):
            r = self.radius * (0.5 + np.random.rand() * 1.0)
            self.landmarks[i, 0] = r * np.cos(angle)
            self.landmarks[i, 1] = r * np.sin(angle)
            self.landmarks[i, 2] = np.random.uniform(-self.height_variation, self.height_variation)
        
        # For each pose, observe the nearby landmarks
        self.observations = []
        max_range = self.radius * 0.8
        
        for i in range(self.n_poses):
            pose_obs = []
            for j, landmark in enumerate(self.landmarks):
                dist = np.linalg.norm(self.ground_truth[i] - landmark)
                if dist < max_range:
                    # Relative observation with noise
                    relative_pos = landmark - self.ground_truth[i]
                    observed_relative = relative_pos + np.random.randn(3) * self.sensor_noise
                    pose_obs.append((j, observed_relative))
            self.observations.append(pose_obs)
    
    def detect_loop_closures(self, trajectory):
        """Detect loop closure when seeing again the same landmark"""
        loop_closures = []
        
        # Find poses that are observing the same landmark
        for i in range(self.n_poses - 30):  # They must not be too close
            for j in range(i + 30, self.n_poses):
                # Check common landmarks
                landmarks_i = set([obs[0] for obs in self.observations[i]])
                landmarks_j = set([obs[0] for obs in self.observations[j]])
                common_landmarks = landmarks_i.intersection(landmarks_j)
                
                if len(common_landmarks) >= 3:  # If at least 3 landmarks are in common
                    # Estimate real distance based on ground truth (should use the landmark, but this is an example)
                    actual_dist = np.linalg.norm(self.ground_truth[i] - self.ground_truth[j])
                    if actual_dist < self.loop_closure_threshold:
                        constraint = self.ground_truth[j] - self.ground_truth[i]
                        loop_closures.append((i, j, constraint))
        
        return loop_closures
    
    def optimize_with_loop_closure(self, initial_trajectory, loop_closures):
        """Optimize the path using pose graph optimization"""
        if not loop_closures:
            return initial_trajectory.copy()
        
        n = self.n_poses
        optimized = initial_trajectory.copy()
        
        # Weights for different links
        w_odom = 1.0      # weight for standard odometry
        w_loop = 100.0    # Weight for loop closure links!
        
        # Simplified iterative optimization
        for iteration in range(50):
            prev_optimized = optimized.copy()
            
            # For each pose (but the first, which is fixed`)
            for i in range(1, n):
                corrections = []
                weights = []
                
                # Odometry from the predecessor
                if i > 0:
                    expected = initial_trajectory[i] - initial_trajectory[i-1]
                    corrections.append(optimized[i-1] + expected)
                    weights.append(w_odom)
                
                # Odometry to the following
                if i < n-1:
                    expected = initial_trajectory[i+1] - initial_trajectory[i]
                    corrections.append(optimized[i+1] - expected)
                    weights.append(w_odom)
                
                # Loop closure links
                for lc_i, lc_j, constraint in loop_closures:
                    if lc_i == i:
                        corrections.append(optimized[lc_j] - constraint)
                        weights.append(w_loop)
                    elif lc_j == i:
                        corrections.append(optimized[lc_i] + constraint)
                        weights.append(w_loop)
                
                # Weighted mean of the corrections
                if corrections:
                    weights = np.array(weights)
                    weights = weights / weights.sum()
                    weighted_correction = np.zeros(3)
                    for corr, w in zip(corrections, weights):
                        weighted_correction += w * corr
                    
                    # Apply the damping factor
                    alpha = 0.5
                    optimized[i] = alpha * weighted_correction + (1-alpha) * optimized[i]
            
            # Check convergence
            change = np.mean(np.linalg.norm(optimized - prev_optimized, axis=1))
            if change < 0.001:
                break
        
        return optimized
    
    def run_slam(self):
        """Run the complete 3D SLAM simulator"""
        print("=" * 60)
        print("3D SLAM SIMULATOR WITH LOOP CLOSURE")
        print("=" * 60)
        print(f"Pose number: {self.n_poses}")
        print(f"Odometry noise: {self.odometry_noise}")
        print(f"Sensor noise: {self.sensor_noise}")
        
        # Pre loop closure (only odometry with drift)
        self.trajectory_pre_loop = self.odometry.copy()
        
        # Check loop closure
        loop_closures = self.detect_loop_closures(self.odometry)
        print(f"\nLoop closures found: {len(loop_closures)}")
        
        if loop_closures:
            print("First 3 loop closures:")
            for i, (idx1, idx2, _) in enumerate(loop_closures[:3]):
                print(f"  {i+1}. Between poses {idx1} and {idx2}")
        
        # Post loop closure (optimized)
        self.trajectory_post_loop = self.optimize_with_loop_closure(
            self.trajectory_pre_loop, loop_closures
        )
        
        # Calcualte errors
        self.calculate_errors()
        
        return loop_closures
    
    def calculate_errors(self):
        """Calcolate errors with respect to ground truth"""
        # Errors pre loop closure
        errors_pre = np.linalg.norm(self.trajectory_pre_loop - self.ground_truth, axis=1)
        error_pre_mean = np.mean(errors_pre)
        error_pre_max = np.max(errors_pre)
        
        # Errors post loop closure  
        errors_post = np.linalg.norm(self.trajectory_post_loop - self.ground_truth, axis=1)
        error_post_mean = np.mean(errors_post)
        error_post_max = np.max(errors_post)
        
        print(f"\n{'='*40}")
        print("ERRORS METRICS:")
        print(f"{'='*40}")
        print(f"Mean error:")
        print(f"  Pre loop closure:  {error_pre_mean:.3f}")
        print(f"  Post loop closure: {error_post_mean:.3f}")
        print(f"  Improvement:     {((error_pre_mean - error_post_mean) / error_pre_mean * 100):.1f}%")
        print(f"\nMax Error:")
        print(f"  Pre loop closure:  {error_pre_max:.3f}")
        print(f"  Post loop closure: {error_post_max:.3f}")
        print(f"  Improvement:     {((error_pre_max - error_post_max) / error_pre_max * 100):.1f}%")
        
        return error_pre_mean, error_post_mean
    
    def _setup_axis(self, ax, title, legend_loc='upper right', is_3d=False, equal_axis=False):
        """Helper function to set up plot aesthetics."""
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X [m]', fontsize=12)
        ax.set_ylabel('Y [m]', fontsize=12)
        if is_3d:
            ax.set_zlabel('Z [m]', fontsize=12)
        ax.legend(loc=legend_loc, fontsize=10)
        ax.grid(True, alpha=0.3)
        if equal_axis:
            ax.axis('equal')

    def _save_plot(self, fig, ax, path, filename, is_3d=False, save_figures=True):
        """Helper function to save plots, with support for 3D views."""
        if not save_figures:
            plt.close(fig)
            return

        if not os.path.exists(path):
            os.makedirs(path)

        if is_3d:
            angles = {'default': (30, 45), 'top': (60, 120), 'side': (5, 90)}
            for name, (elev, azim) in angles.items():
                ax.view_init(elev=elev, azim=azim)
                full_path = os.path.join(path, f"{filename}_{name}.png")
                fig.savefig(full_path, dpi=200, bbox_inches='tight')
                print(f"  Saved: {os.path.basename(full_path)}")
        else:
            full_path = os.path.join(path, f"{filename}.png")
            fig.savefig(full_path, dpi=200, bbox_inches='tight')
            print(f"  Saved: {os.path.basename(full_path)}")
        plt.close(fig)

    def visualize_and_save_results(self, loop_closures, save_figures=True):
        """
        Visualize SLAM results by generating a composite plot and saving 
        individual high-resolution figures.
        """
        output_dir = 'slam_output'
        print("\nGenerating and saving plots...")

        # Individual Plots

        # 1. 3D View: Pre Loop Closure
        fig1, ax1 = plt.subplots(figsize=(10, 8), subplot_kw={'projection': '3d'})
        ax1.plot(self.ground_truth[:, 0], self.ground_truth[:, 1], self.ground_truth[:, 2], 'g-', label='Ground Truth', linewidth=3, alpha=0.8)
        ax1.plot(self.trajectory_pre_loop[:, 0], self.trajectory_pre_loop[:, 1], self.trajectory_pre_loop[:, 2], 'r-', label='Pre Loop Closure', linewidth=2)
        ax1.scatter(self.landmarks[:, 0], self.landmarks[:, 1], self.landmarks[:, 2], c='cyan', marker='^', s=100, alpha=0.6, edgecolors='black', label='Landmarks')
        ax1.scatter(*self.trajectory_pre_loop[0], c='green', s=300, marker='o', label='Start', edgecolors='black', zorder=5)
        ax1.scatter(*self.trajectory_pre_loop[-1], c='red', s=300, marker='s', label='End (Drift)', edgecolors='black', zorder=5)
        self._setup_axis(ax1, 'PRE Loop Closure - 3D View', is_3d=True)
        self._save_plot(fig1, ax1, output_dir, '01_pre_loop_closure_3d', is_3d=True, save_figures=save_figures)

        # 2. 3D View: Post Loop Closure
        fig2, ax2 = plt.subplots(figsize=(10, 8), subplot_kw={'projection': '3d'})
        ax2.plot(self.ground_truth[:, 0], self.ground_truth[:, 1], self.ground_truth[:, 2], 'g-', label='Ground Truth', linewidth=3, alpha=0.8)
        ax2.plot(self.trajectory_post_loop[:, 0], self.trajectory_post_loop[:, 1], self.trajectory_post_loop[:, 2], 'b-', label='Post Loop Closure', linewidth=2)
        ax2.scatter(self.landmarks[:, 0], self.landmarks[:, 1], self.landmarks[:, 2], c='cyan', marker='^', s=100, alpha=0.6, edgecolors='black', label='Landmarks')
        for i, (lc_i, lc_j, _) in enumerate(loop_closures[:8]):
            ax2.plot([self.trajectory_post_loop[lc_i, 0], self.trajectory_post_loop[lc_j, 0]], 
                     [self.trajectory_post_loop[lc_i, 1], self.trajectory_post_loop[lc_j, 1]], 
                     [self.trajectory_post_loop[lc_i, 2], self.trajectory_post_loop[lc_j, 2]], 
                     'orange', alpha=0.5, linewidth=2, linestyle='--', label='Loop Closure' if i == 0 else "")
        ax2.scatter(*self.trajectory_post_loop[0], c='green', s=300, marker='o', label='Start', edgecolors='black', zorder=5)
        ax2.scatter(*self.trajectory_post_loop[-1], c='blue', s=300, marker='s', label='End (Corrected)', edgecolors='black', zorder=5)
        self._setup_axis(ax2, 'POST Loop Closure - 3D View', is_3d=True)
        self._save_plot(fig2, ax2, output_dir, '02_post_loop_closure_3d', is_3d=True, save_figures=save_figures)

        # 3. 3D View: Comparison
        fig3, ax3 = plt.subplots(figsize=(10, 8), subplot_kw={'projection': '3d'})
        ax3.plot(self.ground_truth[:, 0], self.ground_truth[:, 1], self.ground_truth[:, 2], 'g-', label='Ground Truth', linewidth=3, alpha=0.9)
        ax3.plot(self.trajectory_pre_loop[:, 0], self.trajectory_pre_loop[:, 1], self.trajectory_pre_loop[:, 2], 'r--', label='Pre Loop Closure', linewidth=2, alpha=0.7)
        ax3.plot(self.trajectory_post_loop[:, 0], self.trajectory_post_loop[:, 1], self.trajectory_post_loop[:, 2], 'b-', label='Post Loop Closure', linewidth=2, alpha=0.8)
        self._setup_axis(ax3, 'Complete Comparison - 3D View', is_3d=True)
        self._save_plot(fig3, ax3, output_dir, '03_comparison_3d', is_3d=True, save_figures=save_figures)

        # 4. 2D View: Top-Down (XY)
        fig4, ax4 = plt.subplots(figsize=(10, 10))
        ax4.plot(self.ground_truth[:, 0], self.ground_truth[:, 1], 'g-', label='Ground Truth', linewidth=3, alpha=0.9)
        ax4.plot(self.trajectory_pre_loop[:, 0], self.trajectory_pre_loop[:, 1], 'r--', label='Pre Loop Closure', linewidth=2, alpha=0.7)
        ax4.plot(self.trajectory_post_loop[:, 0], self.trajectory_post_loop[:, 1], 'b-', label='Post Loop Closure', linewidth=2, alpha=0.8)
        ax4.scatter(self.landmarks[:, 0], self.landmarks[:, 1], c='cyan', marker='^', s=100, alpha=0.5, edgecolors='black', label='Landmarks')
        ax4.scatter(self.ground_truth[0, 0], self.ground_truth[0, 1], c='green', s=200, marker='o', zorder=5, edgecolors='black', label='Start')
        ax4.scatter(self.trajectory_pre_loop[-1, 0], self.trajectory_pre_loop[-1, 1], c='red', s=200, marker='s', zorder=5, edgecolors='black', label='End Pre-LC')
        ax4.scatter(self.trajectory_post_loop[-1, 0], self.trajectory_post_loop[-1, 1], c='blue', s=200, marker='s', zorder=5, edgecolors='black', label='End Post-LC')
        self._setup_axis(ax4, 'Top-Down View (XY Plane)', legend_loc='best', equal_axis=True)
        self._save_plot(fig4, ax4, output_dir, '04_view_xy', save_figures=save_figures)

        # 5. 2D View: Side (XZ)
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        ax5.plot(self.ground_truth[:, 0], self.ground_truth[:, 2], 'g-', label='Ground Truth', linewidth=3, alpha=0.9)
        ax5.plot(self.trajectory_pre_loop[:, 0], self.trajectory_pre_loop[:, 2], 'r--', label='Pre Loop Closure', linewidth=2, alpha=0.7)
        ax5.plot(self.trajectory_post_loop[:, 0], self.trajectory_post_loop[:, 2], 'b-', label='Post Loop Closure', linewidth=2, alpha=0.8)
        self._setup_axis(ax5, 'Side View (XZ Plane)', legend_loc='best', equal_axis=True)
        ax5.set_ylabel('Z [m]')
        self._save_plot(fig5, ax5, output_dir, '05_view_xz', save_figures=save_figures)

        # 6. Error Plot
        fig6, ax6 = plt.subplots(figsize=(12, 6))
        error_pre = np.linalg.norm(self.trajectory_pre_loop - self.ground_truth, axis=1)
        error_post = np.linalg.norm(self.trajectory_post_loop - self.ground_truth, axis=1)
        ax6.plot(error_pre, 'r-', label='Pre Loop Closure Error', linewidth=2.5, alpha=0.8)
        ax6.plot(error_post, 'b-', label='Post Loop Closure Error', linewidth=2.5, alpha=0.8)
        ax6.fill_between(range(len(error_pre)), 0, error_pre, alpha=0.2, color='red')
        ax6.fill_between(range(len(error_post)), 0, error_post, alpha=0.2, color='blue')
        for i, (lc_i, lc_j, _) in enumerate(loop_closures):
             ax6.axvline(x=lc_j, color='orange', linestyle=':', alpha=0.6, linewidth=1.5, label='Loop Closure' if i == 0 else "")
        self._setup_axis(ax6, 'Trajectory Error Over Time', legend_loc='upper left')
        ax6.set_xlabel('Time Step')
        ax6.set_ylabel('Error [m]')
        self._save_plot(fig6, ax6, output_dir, '06_time_error', save_figures=save_figures)
        
        print("\nAll individual plots saved successfully!")

        # Composite Plot
        print("Generating composite plot...")
        fig_comp = plt.figure(figsize=(20, 12))
        fig_comp.suptitle('3D SLAM with Loop Closure - Full Analysis', fontsize=18, fontweight='bold')
        
        # Row 1: 3D views
        ax_c1 = fig_comp.add_subplot(231, projection='3d')
        ax_c1.plot(self.ground_truth[:, 0], self.ground_truth[:, 1], self.ground_truth[:, 2], 'g-', label='Ground Truth', linewidth=2)
        ax_c1.plot(self.trajectory_pre_loop[:, 0], self.trajectory_pre_loop[:, 1], self.trajectory_pre_loop[:, 2], 'r-', label='Pre-LC', linewidth=1.5)
        self._setup_axis(ax_c1, 'Pre-LC Trajectory', is_3d=True)

        ax_c2 = fig_comp.add_subplot(232, projection='3d')
        ax_c2.plot(self.ground_truth[:, 0], self.ground_truth[:, 1], self.ground_truth[:, 2], 'g-', label='Ground Truth', linewidth=2)
        ax_c2.plot(self.trajectory_post_loop[:, 0], self.trajectory_post_loop[:, 1], self.trajectory_post_loop[:, 2], 'b-', label='Post-LC', linewidth=1.5)
        self._setup_axis(ax_c2, 'Post-LC Trajectory', is_3d=True)

        ax_c3 = fig_comp.add_subplot(233, projection='3d')
        ax_c3.plot(self.ground_truth[:, 0], self.ground_truth[:, 1], self.ground_truth[:, 2], 'g-', label='Ground Truth', linewidth=2)
        ax_c3.plot(self.trajectory_pre_loop[:, 0], self.trajectory_pre_loop[:, 1], self.trajectory_pre_loop[:, 2], 'r--', label='Pre-LC', linewidth=1.5, alpha=0.7)
        ax_c3.plot(self.trajectory_post_loop[:, 0], self.trajectory_post_loop[:, 1], self.trajectory_post_loop[:, 2], 'b-', label='Post-LC', linewidth=1.5)
        self._setup_axis(ax_c3, '3D Comparison', is_3d=True)
        
        # Row 2: 2D views and error
        ax_c4 = fig_comp.add_subplot(234)
        ax_c4.plot(self.ground_truth[:, 0], self.ground_truth[:, 1], 'g-', label='Ground Truth', linewidth=2)
        ax_c4.plot(self.trajectory_pre_loop[:, 0], self.trajectory_pre_loop[:, 1], 'r--', label='Pre-LC', linewidth=1.5, alpha=0.7)
        ax_c4.plot(self.trajectory_post_loop[:, 0], self.trajectory_post_loop[:, 1], 'b-', label='Post-LC', linewidth=1.5)
        self._setup_axis(ax_c4, 'Top-Down View (XY)', equal_axis=True, legend_loc='best')

        ax_c5 = fig_comp.add_subplot(235)
        ax_c5.plot(self.ground_truth[:, 0], self.ground_truth[:, 2], 'g-', label='Ground Truth', linewidth=2)
        ax_c5.plot(self.trajectory_pre_loop[:, 0], self.trajectory_pre_loop[:, 2], 'r--', label='Pre-LC', linewidth=1.5, alpha=0.7)
        ax_c5.plot(self.trajectory_post_loop[:, 0], self.trajectory_post_loop[:, 2], 'b-', label='Post-LC', linewidth=1.5)
        self._setup_axis(ax_c5, 'Side View (XZ)', equal_axis=True, legend_loc='best')
        ax_c5.set_ylabel('Z [m]')

        ax_c6 = fig_comp.add_subplot(236)
        ax_c6.plot(error_pre, 'r-', label='Error Pre-LC', linewidth=2, alpha=0.8)
        ax_c6.plot(error_post, 'b-', label='Error Post-LC', linewidth=2, alpha=0.8)
        ax_c6.fill_between(range(len(error_pre)), 0, error_pre, alpha=0.2, color='red')
        ax_c6.fill_between(range(len(error_post)), 0, error_post, alpha=0.2, color='blue')
        self._setup_axis(ax_c6, 'Position Error Over Time', legend_loc='upper left')
        ax_c6.set_xlabel('Time Step')
        ax_c6.set_ylabel('Error [m]')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_figures:
            composite_path = os.path.join(output_dir, '00_full_analysis.png')
            fig_comp.savefig(composite_path, dpi=150, bbox_inches='tight')
            print(f"  Saved composite plot: {os.path.basename(composite_path)}")

        plt.show()


if __name__ == "__main__":
    # Create and run the simulator
    np.random.seed(42)  # For seeded results
    slam_sim = SLAMSimulator3D(n_poses=100, radius=10, height_variation=2)
    
    # Run the SLAM
    loop_closures = slam_sim.run_slam()
    
    # Visualize the results and save plots
    slam_sim.visualize_and_save_results(loop_closures, save_figures=True)
    
    # Print final drift statistics
    print(f"\n{'='*50}")
    print("FINAL DRIFT ANALYSIS")
    print(f"{'='*50}")
    drift_pre = np.linalg.norm(slam_sim.trajectory_pre_loop[-1] - slam_sim.ground_truth[0])
    drift_post = np.linalg.norm(slam_sim.trajectory_post_loop[-1] - slam_sim.ground_truth[0])
    print(f"Distance from start point (should be ~0):")
    print(f"  Ground truth:      {np.linalg.norm(slam_sim.ground_truth[-1] - slam_sim.ground_truth[0]):.3f}")
    print(f"  Pre loop closure:  {drift_pre:.3f}")
    print(f"  Post loop closure: {drift_post:.3f}")