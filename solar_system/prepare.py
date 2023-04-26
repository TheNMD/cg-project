import cv2

def prepare_texture(in_files, out_file):
    images = [cv2.imread(file) for file in in_files]
    texture = cv2.hconcat(images)
    cv2.imwrite(out_file, texture)

in_files = ["./image/earth.jpg", "./image/moon.jpg", "./image/sun.jpg"]
out_files = "./image/solar_system.jpg"
prepare_texture(in_files, out_files)