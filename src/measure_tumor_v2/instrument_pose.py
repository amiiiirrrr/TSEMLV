import numpy as np
import cv2
import os
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from skimage.morphology import skeletonize
from sklearn.linear_model import LinearRegression
import pdb

# --- Module: Estimate Instrument Pose ---
# def estimate_instrument_pose(K_matrix, instrument_radius_mm, line1_coeffs, line2_coeffs, viz_copy):
#     """
#     Estimates the 3D pose of a cylindrical instrument...
#     """
#     if line1_coeffs is None or line2_coeffs is None:
#         print("Debug (estimate_pose): Input line coefficients are None.")
#         return None, None

#     l1 = np.array(line1_coeffs).reshape(3, 1)
#     l2 = np.array(line2_coeffs).reshape(3, 1)
#     K = K_matrix
#     r = instrument_radius_mm

#     K = K.astype(np.float64)
#     l1 = l1.astype(np.float64)
#     l2 = l2.astype(np.float64)

#     try: # Added try-except for potential matrix operation errors
#         n1 = K.T @ l1
#         n2 = K.T @ l2
#         n1 = n1.flatten()
#         n2 = n2.flatten()
#     except Exception as e:
#             print(f"Debug (estimate_pose): Error during K.T @ l calculation: {e}")
#             return None, None


#     n1_norm = np.linalg.norm(n1)
#     n2_norm = np.linalg.norm(n2)

#     if n1_norm < 1e-9 or n2_norm < 1e-9:
#         print("Debug (estimate_pose): Error: Normal vector for a line is close to zero.")
#         return None, None

#     n1_unit = n1 / n1_norm
#     n2_unit = n2 / n2_norm

#     v_L = np.cross(n1_unit, n2_unit)
#     v_L_norm = np.linalg.norm(v_L)

#     if v_L_norm < 1e-9:
#         print("Debug (estimate_pose): Error: Back-projection planes are parallel (v_L norm zero).")
#         return None, None
#     v_L_unit = v_L / v_L_norm

#     n_m_direction = n1_unit + n2_unit
#     n_m_direction_norm = np.linalg.norm(n_m_direction)

#     if n_m_direction_norm < 1e-9:
#         print("Debug (estimate_pose): Error: Bisector plane normal is close to zero.")
#         return None, None
#     n_m_unit = n_m_direction / n_m_direction_norm

#     v_OPc_direction = np.cross(v_L_unit, n_m_unit)
#     v_OPc_direction_norm = np.linalg.norm(v_OPc_direction)

#     if v_OPc_direction_norm < 1e-9:
#         print("Debug (estimate_pose): Error: Direction vector v_OPc is close to zero.")
#         return None, None
#     v_OPc_unit = v_OPc_direction / v_OPc_direction_norm

#     cos_alpha = np.clip(np.dot(n1_unit, n2_unit), -1.0, 1.0)
#     # print('cos_alpha', cos_alpha)
#     alpha = np.arccos(cos_alpha)
#     # print('alpha', alpha)

#     if alpha < 1e-6 or np.abs(alpha - np.pi) < 1e-6:
#         print("Debug (estimate_pose): Error: Angle alpha between planes is too small or too large.")
#         return None, None

#     sin_alpha_half = np.sin(alpha / 2.0)
#     # print('sin_alpha_half', sin_alpha_half)
#     if np.abs(sin_alpha_half) < 1e-9:
#         print("Debug (estimate_pose): Error: sin(alpha/2) is close to zero.")
#         return None, None

#     d_OPc = r / sin_alpha_half
#     P_c = d_OPc * v_OPc_unit

#     if P_c[2] < 0:
#             print(f"Debug (estimate_pose): Warning: Estimated Z-depth of P_c is negative ({P_c[2]:.3f}). Flipping sign.")
#             P_c = -P_c # Flip to be in front of the camera

#     # pdb.set_trace()
#     return P_c, v_L_unit, viz_copy

def estimate_instrument_pose(K_matrix, instrument_radius_mm, line1_coeffs, line2_coeffs, viz_copy):
    """
    Estimates the 3D pose of a cylindrical instrument...
    Returns (P_c, v_L_unit, viz_copy), or (None, None, viz_copy) on failure.
    """
    # --- input checks ---
    if line1_coeffs is None or line2_coeffs is None:
        print("Debug: Input line coefficients are None.")
        return None, None, viz_copy

    r = float(instrument_radius_mm)
    if r <= 0:
        print("Debug: instrument_radius_mm must be > 0.")
        return None, None, viz_copy

    # --- prepare data ---
    l1 = np.array(line1_coeffs, dtype=np.float64).reshape(3, 1)
    l2 = np.array(line2_coeffs, dtype=np.float64).reshape(3, 1)
    K  = np.array(K_matrix, dtype=np.float64)

    n1 = (K.T @ l1).flatten()
    n2 = (K.T @ l2).flatten()

    # --- normalize and guard against degenerate cases ---
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-9 or n2_norm < 1e-9:
        print("Debug: One of the plane normals is zero-length.")
        return None, None, viz_copy

    n1_unit = n1 / n1_norm
    n2_unit = n2 / n2_norm
    # print("n1_unit:", n1_unit, " n2_unit:", n2_unit, " dot:", np.dot(n1_unit,n2_unit))

    # --- disambiguate sign so they form an acute angle ---
    if float(n1_unit.dot(n2_unit)) < 0:
        n2_unit = -n2_unit

    # --- intersection line direction ---
    v_L = np.cross(n1_unit, n2_unit)
    v_L_norm = np.linalg.norm(v_L)
    if v_L_norm < 1e-9:
        print("Debug: Back-projection planes are parallel.")
        return None, None, viz_copy
    v_L_unit = v_L / v_L_norm
    # print("v_L_unit:", v_L_unit, "norm:", v_L_norm)

    # --- bisector plane normal ---
    n_m = n1_unit + n2_unit
    n_m_norm = np.linalg.norm(n_m)
    if n_m_norm < 1e-9:
        print("Debug: Bisector plane normal is close to zero.")
        return None, None, viz_copy
    n_m_unit = n_m / n_m_norm
    # print("n_m_unit:", n_m_unit, "norm:", n_m_norm)

    # --- direction to circle center ---
    v_OPc = np.cross(v_L_unit, n_m_unit)
    v_OPc_norm = np.linalg.norm(v_OPc)
    if v_OPc_norm < 1e-9:
        print("Debug: Direction vector to P_c is zero.")
        return None, None, viz_copy
    v_OPc_unit = v_OPc / v_OPc_norm

    # --- angle between planes and depth ---
    cos_alpha = np.clip(np.dot(n1_unit, n2_unit), -1, 1)
    alpha     = np.arccos(cos_alpha)
    # print("alpha [deg]:", np.degrees(alpha))

    # --- new: clamp very small angles instead of bailing out ---
    eps = 0.005  # ~0.3°
    if alpha < eps:
        print(f"Debug: α ({alpha:.6f}) < eps, clamping to {eps:.6f}")
        alpha = eps

    sin_half = np.sin(alpha/2.0)
    # print("sin(alpha/2):", sin_half)
    if abs(sin_half) < 1e-9:
        # this really should never happen now that we've clamped
        print("Debug: sin(alpha/2) too small even after clamp.")
        return None, None, viz

    d_OPc = r / sin_half
    P_c   = d_OPc * v_OPc_unit

    if P_c[2] < 0:
        print(f"Debug: Z negative ({P_c[2]:.3f}), flipping.")
        P_c = -P_c

    return P_c, v_L_unit, viz_copy


def fit_shaft_lines(segmentation_mask, original_image_bgr, path_save):

    contours, _ = cv2.findContours(segmentation_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No contours found.")
        return None
    contour = max(contours, key=cv2.contourArea)
    contour = contour[:, 0, :]

    all_pts = np.vstack(contour).squeeze()

    # Skeletonize
    skel = skeletonize(segmentation_mask)
    
    # Collect skeleton points
    skel_coords = np.column_stack(np.where(skel > 0))  # (row, col)
    if len(skel_coords) < 10:
        print("Too few skeleton points.")
        return None
    skel_pts = skel_coords[:, ::-1].astype(float)  # (x, y)
    # PCA on skeleton to find main axis (shaft direction)
    skel_mean = np.mean(skel_pts, axis=0)
    skel_centered = skel_pts - skel_mean
    _, _, vh = np.linalg.svd(skel_centered)
    shaft_dir = vh[0]
    ortho_dir = np.array([-shaft_dir[1], shaft_dir[0]])

    # For the line fitting of the shaft, filter out points close the image edges:
    # Keep skipping boundary edges for now, as they are less likely to be part of the straight shaft
    atol = 5 # remove points within 5 pixels of border
    h, w = segmentation_mask.shape[:2]
    # close_to_boundary = np.array([np.any(np.isclose(point, [0, 0], atol=atol)) or np.any(np.isclose(point, [w-1, h-1], atol=atol)) for point in contour])
    close_to_boundary = np.array([np.any(np.isclose(point, [32, 5], atol=atol)) or np.any(np.isclose(point, [w-1-32, h-1], atol=atol)) for point in contour])
    pts = all_pts[~close_to_boundary]

    # Determine centered points of the contour (w.r.t. the center of the skeleton points)
    centered = pts - skel_mean

    shaft_coords = centered @ shaft_dir
    ortho_coords = centered @ ortho_dir
    shaft_bins = np.round(shaft_coords).astype(int)

    side1_pts = np.array([point for point, ortho_coord in zip(pts, ortho_coords) if ortho_coord >= 0])
    side2_pts = np.array([point for point, ortho_coord in zip(pts, ortho_coords) if ortho_coord < 0])

    # line_1_coeffs = fit_line_ransac_robust(side1_pts)
    # line_2_coeffs = fit_line_ransac_robust(side2_pts)

    line_1_coeffs = fit_line_ransac_prior(side1_pts, shaft_direction=shaft_dir)
    line_2_coeffs = fit_line_ransac_prior(side2_pts, shaft_direction=shaft_dir)

    # line_1_coeffs = fit_line_theilsen_robust(side1_pts)
    # line_2_coeffs = fit_line_theilsen_robust(side2_pts)

    # Visualization
    viz_copy = original_image_bgr.copy()
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(viz_copy, [box], 0, (0, 0, 255), 1)

    for px, py in side1_pts:
        cv2.circle(viz_copy, (int(px), int(py)), 3, (255, 55, 0), -1)
    for px, py in side2_pts:
        cv2.circle(viz_copy, (int(px), int(py)), 3, (100, 150, 10), -1)

    h, w = viz_copy.shape[:2]
    for coeffs in [line_1_coeffs, line_2_coeffs]:
        if coeffs is not None and len(coeffs) == 3:
            a, b, c = coeffs
            pt1, pt2 = get_line_boundary_points(a, b, c, w, h)
            if pt1 is not None and pt2 is not None:
                cv2.line(viz_copy, pt1, pt2, (0, 255, 0), 2)

    pc_x, pc_y = int(skel_mean[0]), int(skel_mean[1])
    pc_xn, pc_yn = int((skel_mean + 50 * shaft_dir)[0]), int((skel_mean + 50 * shaft_dir)[1])
    cv2.line(viz_copy, (pc_x, pc_y), (pc_xn, pc_yn), (10, 150, 50), 2)

    # output_path = f"{self.args.output_dir}/{self.image_name}/{self.image_name}_intermediate.png"
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # cv2.imwrite(output_path, viz_copy)

    return line_1_coeffs, line_2_coeffs, viz_copy


def fit_line_ransac_prior(points, shaft_direction=None):
    """
    Fits a line to a set of 2D points using RANSAC, optionally incorporating
    a prior shaft direction to guide axis swapping.

    Args:
        points (np.ndarray): A NumPy array of shape (n_points, 2) representing the 2D points.
        shaft_direction (np.ndarray or None): A 2-element NumPy array [vx, vy]
                                                representing the approximate direction of the shaft.
                                                If None, axis swapping is based on data variance
                                                (original RANSAC logic).

    Returns:
        np.ndarray or None: A NumPy array containing the normalized [a, b, c] coefficients
                            for the line ax + by + c = 0, or None if fitting fails.
    """
    if len(points) < 2:
        return None # Cannot fit a line with fewer than 2 points

    swap_axes = False
    if shaft_direction is not None and len(shaft_direction) == 2:
        # Use shaft direction to decide axis swapping
        vx, vy = shaft_direction
        # If the absolute value of the y component of the shaft direction
        # is greater than the absolute value of the x component,
        # the shaft is more vertical, so we should regress x on y.
        if np.abs(vy) > np.abs(vx):
                swap_axes = True
    else:
        # Fallback to variance check if no valid shaft_direction is provided.
        # This is the original RANSAC logic.
        variance_x = np.var(points[:, 0])
        variance_y = np.var(points[:, 1])
        if variance_y < variance_x * 0.1: # If significantly less variance in y than in x
                swap_axes = True # Data is likely mostly horizontal or vertical, regress x on y


    try:
        if not swap_axes:
            # Try fitting y = mx + b
            X = points[:, 0].reshape(-1, 1) # Independent variable
            y = points[:, 1]              # Dependent variable

            # RANSAC parameters: max_trials, min_samples (minimum points to fit the model),
            # residual_threshold (max distance for a point to be an inlier).
            # These may need tuning based on the expected noise level.
            ransac = RANSACRegressor(estimator=LinearRegression(), min_samples=2, residual_threshold=1.0) # Tunable parameters
            ransac.fit(X, y)

            if ransac.estimator_ is None:
                # RANSAC failed to find a valid model
                print("RANSAC (y=mx+b) estimator is None.")
                return None

            m = ransac.estimator_.coef_[0]     # Slope
            b_intercept = ransac.estimator_.intercept_ # Y-intercept

            # Convert y = mx + b to ax + by + c = 0
            # mx - y + b = 0
            a = m
            b = -1
            c = b_intercept

        else:
            # Try fitting x = m'y + b'
            X = points[:, 1].reshape(-1, 1) # Independent variable (y)
            y = points[:, 0]              # Dependent variable (x)

            # Use appropriate residual threshold for swapped axes if needed, or keep consistent
            ransac = RANSACRegressor(estimator=LinearRegression(), min_samples=2, residual_threshold=2.0, random_state=38) # Tunable parameters
            ransac.fit(X, y)

            if ransac.estimator_ is None:
                print("RANSAC (x=my+b) estimator is None.")
                return None

            m_prime = ransac.estimator_.coef_[0] # Slope for x = m'y + b'
            b_prime = ransac.estimator_.intercept_ # X-intercept

            # Convert x = m'y + b' to ax + by + c = 0
            # x - m'y - b' = 0
            a = 1
            b = -m_prime
            c = -b_prime

        # Normalize coefficients
        norm = np.sqrt(a**2 + b**2)
        if norm == 0:
                # Should not happen with a valid line fit
                return None

        # Ensure the 'a' coefficient is non-negative for consistent normalization,
        # unless it's a vertical line where 'b' is close to zero.
        if a < 0 and not np.isclose(b, 0):
                a, b, c = -a, -b, -c

        norm = np.sqrt(a**2 + b**2) # Recalculate norm after potential sign change
        return np.array([a / norm, b / norm, c / norm])

    except Exception as e:
        print(f"RANSAC fitting failed: {e}")
        return None

def get_line_boundary_points(a, b, c, w, h):
    """
    Finds two points on the image boundary (w, h) for a line ax + by + c = 0.
    Returns None if line is outside bounds or invalid.
    """
    points = []
    # Check intersection with image borders x=0, x=w, y=0, y=h

    # Intersection with x = 0 (left border)
    if b != 0:
        y0 = -c / b
        if 0 <= y0 <= h:
            points.append((0, int(y0)))

    # Intersection with x = w (right border)
    if b != 0:
        yw = (-a * w - c) / b
        if 0 <= yw <= h:
            points.append((w, int(yw)))

    # Intersection with y = 0 (top border)
    if a != 0:
        x0 = -c / a
        if 0 <= x0 <= w:
            points.append((int(x0), 0))

    # Intersection with y = h (bottom border)
    if a != 0:
        xh = (-b * h - c) / a
        if 0 <= xh <= w:
            points.append((int(xh), h))

    # Remove duplicate points if any (e.g., corners)
    unique_points = []
    for p in points:
        if p not in unique_points:
            unique_points.append(p)

    # We need exactly two distinct points to draw a line segment spanning the image
    if len(unique_points) >= 2:
        # Return the first two points (they will be distinct boundary points)
        return unique_points[0], unique_points[1]
    elif len(unique_points) == 1:
            # Handle case where the line passes through only one boundary point (e.g. tangent)
            # This is tricky, maybe extend it along its direction? For now, return None.
            return None, None
    else:
            # No intersection with boundary (line outside image or a point)
            return None, None