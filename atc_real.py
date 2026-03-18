import cv2
import numpy as np


def extract_body_measurements(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    # Largest contour = animal
    contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(contour)

    # Body measurements
    body_length = float(w)
    body_height = float(h)

    # Chest width (middle slice)
    mid_y = int(y + h // 2)

    if mid_y >= edges.shape[0]:
        mid_y = edges.shape[0] - 1

    row = edges[mid_y]

    chest_pixels = np.where(row > 0)[0]
    if len(chest_pixels) > 0:
        chest_width = float(chest_pixels[-1] - chest_pixels[0])
    else:
        chest_width = float(w * 0.3)

    # Rump angle calculation (FINAL FIX)
    pts = contour.squeeze()

    if len(pts.shape) < 2:
        rump_angle = 20.0
    else:
        top_points = pts[pts[:, 1] < (y + h * 0.3)]

        if len(top_points) > 2:
            vx, vy, x0, y0 = cv2.fitLine(
                top_points, cv2.DIST_L2, 0, 0.01, 0.01
            )

            # ✅ FINAL FIX (extract scalar safely)
            vx = vx.item()
            vy = vy.item()

            angle = np.degrees(np.arctan2(vy, vx))
            rump_angle = float(abs(angle))
        else:
            rump_angle = 20.0

    return {
        "body_length": body_length,
        "height": body_height,
        "chest_width": chest_width,
        "rump_angle": rump_angle
    }


def calculate_atc_score(measurements):

    if measurements is None:
        return {"error": "No animal detected"}

    length_score = float(min(measurements["body_length"] / 10, 20))
    height_score = float(min(measurements["height"] / 10, 20))
    chest_score = float(min(measurements["chest_width"] / 10, 20))
    rump_score = float(min(measurements["rump_angle"], 20))

    body_condition = (length_score + height_score) / 2

    total = (
        length_score
        + height_score
        + chest_score
        + rump_score
        + body_condition
    )

    return {
        "Body Length": round(length_score, 2),
        "Height": round(height_score, 2),
        "Chest Width": round(chest_score, 2),
        "Rump Angle": round(rump_score, 2),
        "Body Condition": round(body_condition, 2),
        "Total Score": round(total, 2),
    }