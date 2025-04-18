import cv2
import numpy as np

def add_noise(image):
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def overlay_image_alpha(img, overlay, pos):
    x, y = pos
    h, w = overlay.shape[:2]

    x1, y1 = max(x - w // 2, 0), max(y - h // 2, 0)
    x2, y2 = min(x1 + w, img.shape[1]), min(y1 + h, img.shape[0])

    overlay_crop = overlay[0:(y2 - y1), 0:(x2 - x1)]
    if overlay_crop.shape[0] == 0 or overlay_crop.shape[1] == 0:
        return img

    b, g, r, a = cv2.split(overlay_crop)
    overlay_rgb = cv2.merge((b, g, r))
    mask = cv2.merge((a, a, a)) / 255.0

    roi = img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = (1.0 - mask) * roi + mask * overlay_rgb

    return img


marker_img = cv2.imread("images/variant-5.jpg")
marker_img = add_noise(marker_img)
cv2.imwrite("noisy_marker.jpg", marker_img)

fly = cv2.imread("fly64.png", cv2.IMREAD_UNCHANGED)


orb = cv2.ORB_create()
kp_marker, des_marker = orb.detectAndCompute(marker_img, None)

cap = cv2.VideoCapture(0)
frame_width = 1280
frame_height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    kp_frame, des_frame = orb.detectAndCompute(frame, None)

    if des_frame is not None and len(des_frame) >= 2:
        matches = bf.match(des_marker, des_frame)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > 10:
            src_pts = np.float32([kp_marker[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                h, w = marker_img.shape[:2]
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                center = np.mean(dst, axis=0).astype(int).ravel()
                cx, cy = center

                color = (0, 255, 0)
                if cx <= 50 and cy <= 50:
                    color = (255, 0, 0)
                elif cx >= frame.shape[1] - 50 and cy >= frame.shape[0] - 50:
                    color = (0, 0, 255)

                frame = cv2.polylines(frame, [np.int32(dst)], True, color, 3, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 5, color, -1)

                frame = overlay_image_alpha(frame, fly, (cx, cy))

    cv2.imshow("Tracking + Fly", frame)
    cv2.namedWindow("Tracking + Fly", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking + Fly", frame_width, frame_height)
    cv2.moveWindow("Tracking + Fly", (1366 - frame_width) // 2, (768 - frame_height) // 2)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
