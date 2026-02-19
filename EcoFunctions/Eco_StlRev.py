import numpy as np
import matplotlib.pyplot as plt
import os

def generate_revolution_solid_stl(points, filename="revolution_solid.stl", num_segments=36, z_subdivisions=4, visualize=False, save_plot_path=None, plot_filename="profile_plot_subdivided.png", height_threshold=2.0, min_subdivisions=2):
    """
    Generates a solid of revolution around Z-axis from 3D points and saves as STL.
    Now with configurable Z-axis subdivision for higher element density and adaptive subdivision.
    
    Parameters:
    -----------
    points : array_like
        Points in format [(x1,y1,z1), (x2,y2,z2), ...] (minimum 3 points)
    filename : str
        STL file name
    num_segments : int
        Revolution segments (more = smoother)
    z_subdivisions : int
        Number of subdivisions per Z-axis segment (more = higher density in Z direction)
    visualize : bool
        Show plot
    save_plot_path : str
        Directory to save profile plot
    plot_filename : str
        Custom filename for the profile plot (default: "profile_plot_subdivided.png")
    height_threshold : float
        Height threshold below which to reduce subdivisions (default: 2.0m)
    min_subdivisions : int
        Minimum subdivisions for short segments (default: 2)
    
    Returns:
    --------
    dict : Solid information
    """
    
    points = np.array(points)
    
    if points.shape[1] != 3:
        raise ValueError("Points must have 3 coordinates (x, y, z)")
    if len(points) < 3:
        raise ValueError("Minimum 3 points required")
    
    def _subdivide_profile_segments(profile, subdivisions_per_segment, height_threshold=2.0, min_subdivisions=2):
        """
        Subdivide each linear segment of the profile with adaptive subdivisions based on segment height
        
        Parameters:
        -----------
        profile : array
            Profile points
        subdivisions_per_segment : int
            Default number of subdivisions
        height_threshold : float
            Height threshold below which to reduce subdivisions (default: 2.0m)
        min_subdivisions : int
            Minimum subdivisions for short segments (default: 2)
        """
        if len(profile) < 2:
            return profile
            
        # Check if profile is closed (remove duplicate closing point temporarily)
        is_closed = np.allclose(profile[0], profile[-1], atol=1e-6)
        working_profile = profile[:-1] if is_closed else profile
        
        if len(working_profile) < 2:
            return profile
            
        subdivided_points = []
        
        # Process each segment between consecutive points
        for i in range(len(working_profile)):
            # Add current point
            subdivided_points.append(working_profile[i])
            
            # Get next point (wrap around for closed profiles)
            next_i = (i + 1) % len(working_profile)
            if next_i == 0 and not is_closed:
                break  # Don't process last segment if profile is open
                
            current_point = working_profile[i]
            next_point = working_profile[next_i]
            
            # Calculate segment height (absolute difference in Z direction)
            segment_height = abs(next_point[1] - current_point[1])  # profile is [radius, height]
            
            # Adaptive subdivision based on segment height
            if segment_height <= height_threshold:
                # Use minimum subdivisions for short segments
                actual_subdivisions = min_subdivisions
                segment_type = "SHORT"
            else:
                # Use full subdivisions for tall segments
                actual_subdivisions = subdivisions_per_segment
                segment_type = "TALL"
            
            # Debug output for subdivision decisions
            print(f"   Segment {i+1}: Height={segment_height:.2f}m → {actual_subdivisions} subdivisions ({segment_type})")
            
            # Create subdivisions between current and next point
            for j in range(1, actual_subdivisions):
                t = j / actual_subdivisions
                interpolated_point = current_point + t * (next_point - current_point)
                subdivided_points.append(interpolated_point)
        
        subdivided_profile = np.array(subdivided_points)
        
        # Close the profile if it was originally closed
        if is_closed:
            subdivided_profile = np.vstack([subdivided_profile, subdivided_profile[0]])
            
        return subdivided_profile
    
    def _generate_closed_profile(pts, z_subdivisions, height_threshold, min_subdivisions):
        """Generate closed profile from points with adaptive Z-axis subdivision"""
        # Sort by Z coordinate
        sorted_indices = np.argsort(pts[:, 2])
        sorted_points = pts[sorted_indices]
        
        # Calculate radius from Z-axis: r = sqrt(x² + y²)
        radii = np.sqrt(sorted_points[:, 0]**2 + sorted_points[:, 1]**2)
        heights = sorted_points[:, 2]
        
        profile = np.column_stack([radii, heights])
        
        # Close profile if needed
        if not np.allclose(profile[0], profile[-1], atol=1e-6):
            profile = np.vstack([profile, profile[0]])
        
        # Subdivide each segment with adaptive logic
        print(f"🔍 Analyzing segments for adaptive subdivision (threshold: {height_threshold}m):")
        subdivided_profile = _subdivide_profile_segments(profile, z_subdivisions, height_threshold, min_subdivisions)
        
        return subdivided_profile
    
    def _create_revolution_solid(profile, num_seg):
        """Create revolution solid vertices and triangles"""
        vertices = []
        triangles = []
        
        angles = np.linspace(0, 2*np.pi, num_seg + 1)
        
        # Generate vertices
        for i, angle in enumerate(angles[:-1]):
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            
            for j, point in enumerate(profile):
                radius, height = point
                x = radius * cos_a
                y = radius * sin_a
                z = height
                vertices.append([x, y, z])
        
        vertices = np.array(vertices)
        
        # Generate triangles
        num_profile_points = len(profile)
        
        for i in range(num_seg):
            i_next = (i + 1) % num_seg
            
            for j in range(num_profile_points - 1):
                v1 = i * num_profile_points + j
                v2 = i * num_profile_points + (j + 1)
                v3 = i_next * num_profile_points + j
                v4 = i_next * num_profile_points + (j + 1)
                
                triangles.append([v1, v2, v3])
                triangles.append([v2, v4, v3])
        
        return vertices, np.array(triangles)
    
    def _calculate_normal(v1, v2, v3):
        """Calculate triangle normal"""
        edge1 = v2 - v1
        edge2 = v3 - v1
        normal = np.cross(edge1, edge2)
        norm_magnitude = np.linalg.norm(normal)
        if norm_magnitude > 1e-10:
            return normal / norm_magnitude
        else:
            return np.array([0, 0, 1])
    
    def _save_stl(vertices, triangles, fname):
        """Save solid in STL format"""
        with open(fname, 'w') as f:
            f.write("solid revolution_solid\n")
            
            for tri in triangles:
                v1, v2, v3 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
                normal = _calculate_normal(v1, v2, v3)
                
                f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
                f.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
                f.write(f"      vertex {v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            
            f.write("endsolid revolution_solid\n")
    
    def _visualize_profile_and_solid(profile, vertices, triangles, original_profile, save_path=None, custom_filename="profile_plot_subdivided.png"):
        """Visualize 2D profile showing original and subdivided points with custom filename"""
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Plot subdivided profile
        ax.plot(profile[:, 0], profile[:, 1], 'g-', linewidth=1.5, alpha=0.8, 
                label=f'Subdivided profile ({len(profile)} points)')
        ax.plot(profile[:, 0], profile[:, 1], 'go', markersize=2, alpha=0.6)
        
        # Plot original profile points
        ax.plot(original_profile[:, 0], original_profile[:, 1], 'ro-', 
                linewidth=3, markersize=8, label=f'Original points ({len(original_profile)} points)')
        
        ax.set_xlabel('Radius (distance from Z-axis) [m]')
        ax.set_ylabel('Height (Z) [m]')
        ax.set_title(f'2D Profile for Z-axis Revolution\nOriginal: {len(original_profile)} points → Subdivided: {len(profile)} points')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis('equal')
        
        plt.tight_layout()
        
        if save_path:
            full_plot_path = os.path.join(save_path, custom_filename)
            plt.savefig(full_plot_path, dpi=300, bbox_inches='tight')
            print(f"   📊 Profile plot saved: {custom_filename}")
        
        if visualize:
            plt.show()
        else:
            plt.close()  # Close figure to free memory
    
    # Generate original profile for comparison
    sorted_indices = np.argsort(points[:, 2])
    sorted_points = points[sorted_indices]
    radii = np.sqrt(sorted_points[:, 0]**2 + sorted_points[:, 1]**2)
    heights = sorted_points[:, 2]
    original_profile = np.column_stack([radii, heights])
    if not np.allclose(original_profile[0], original_profile[-1], atol=1e-6):
        original_profile = np.vstack([original_profile, original_profile[0]])
    
    # Generate subdivided profile and solid using the configurable z_subdivisions parameter with adaptive logic
    profile = _generate_closed_profile(points, z_subdivisions, height_threshold, min_subdivisions)
    vertices, triangles = _create_revolution_solid(profile, num_segments)
    
    # Save STL
    _save_stl(vertices, triangles, filename)
    
    # Visualize if requested with custom filename
    if visualize or save_plot_path:
        _visualize_profile_and_solid(profile, vertices, triangles, original_profile, 
                                   save_path=save_plot_path, custom_filename=plot_filename)
    
    # Get file size
    file_size = os.path.getsize(filename) if os.path.exists(filename) else 0
    
    return {
        'filename': filename,
        'num_vertices': len(vertices),
        'num_triangles': len(triangles),
        'file_size_bytes': file_size,
        'num_original_points': len(points),
        'num_profile_points': len(profile),
        'subdivision_factor': z_subdivisions,
        'vertices': vertices,
        'triangles': triangles,
        'profile': profile,
        'original_points': points,
        'original_profile': original_profile
    }


if __name__ == "__main__":
    # Example usage - same input as before
    P = np.array([
        [3, 0, -2],   # P1
        [5, 0, -2],   # P2
        [10, 0, -1],  # P3
        [10, 0, 3],   # P4
        [3, 0, 3]     # P5
    ])
    
    result = generate_revolution_solid_stl(
        points=P,
        filename="test_solid_subdivided.stl",
        num_segments=36,
        z_subdivisions=6,  # Now configurable!
        visualize=True,
        plot_filename="example_profile.png"  # Custom filename!
    )
    
    print(f"STL created: {result['filename']}")
    print(f"Original points: {result['num_original_points']}")
    print(f"Subdivided profile points: {result['num_profile_points']}")
    print(f"Z subdivisions per segment: {result['subdivision_factor']}")
    print(f"Vertices: {result['num_vertices']}")
    print(f"Triangles: {result['num_triangles']}")
