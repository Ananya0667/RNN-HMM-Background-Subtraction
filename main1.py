from background_substraction import BackgroundSubtraction
import numpy as np
import cv2
import os
import random
import math

input_video=cv2.VideoCapture("D:\ELL784_Assignment_1/umcp.mpg")

success, bgr_image = input_video.read()
count_of_frames = 0
#frame read succesfully--success
while success:
  # Convert the bgr_image to grayscale_image
  gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
  cv2.imwrite("D:\ELL784_Assignment_1\Grayscale_Images/frame_gray%d.jpg" % count_of_frames,gray_image)     # save frame as .jpg file
  # bgr_image contains the frame in BGR format
  success,bgr_image = input_video.read()
  count_of_frames += 1

print(count_of_frames+1)

count_of_frames = 0
# Process each image in the folder
for filename in os.listdir("D:\ELL784_Assignment_1\Grayscale_Images/"):
    if filename.endswith(".jpg"):
        img1 = cv2.imread("D:\ELL784_Assignment_1\Grayscale_Images/frame_gray%d.jpg" %count_of_frames)
        # Extract grayscale value
        v = img1[:, :]

        # cv2.imwrite("D:\ELL784_Assignment_1\Grayscale_Images/frame_gray%d.jpg" % count_of_frames,v)  # Save frame as .jpg file
        count_of_frames += 1
height, width = img1.shape[:2]

print(f"Processed {count_of_frames} images")
print(height, width)

G_comp = []
for y in range(height):             #(height):
  G_comp.append([])
  for x in range(width):           #(width):
    G_comp[y].append([])

def gaussian_model(x, k):
    mean = []
    var = []
    weights = []

    sum = 0
    for i in range(k):
        mean.append(random.randint(0, 255))
        var.append(random.randint(1, 1000))
        weights.append(round(random.uniform(0, 1), 3))
        sum += weights[i]

    sum1 = 0
    for i in range(k):
        weights[i] = weights[i] / sum

    itr = 30

    #print(weights)
    #print(f"var={var}")

    def normal_funct(x, mean, variance):
        power_term = (((x - mean) ** 2) / variance) / 2
        numerator = math.exp((-1) * power_term)
        denominator = math.sqrt(variance * 2 * math.pi)
        value = numerator / denominator
        return value

    N = len(x)
    for h in range(itr):
        # ####E Step:
        resp = []
        for i in range(len(x)):
            sum = 0
            for j in range(k):
                sum += weights[j] * normal_funct(x[i], mean[j], var[j])
            r = []
            for j in range(k):
                b = (weights[j] * normal_funct(x[i], mean[j], var[j])) / sum
                r.append(b)
            ##### Responsibility per data
            resp.append(r)
        ####### N_j per compnent
        sum2 = 0
        N_j = []
        for j in range(k):
            for i in range(len(x)):
                sum2 += resp[i][j]

            if sum2 < 10**-20:
                sum2 = 10**-19
            N_j.append(sum2)
        ####### M-step
        updated_MU = []
        updated_var = []
        updated_weights = []
        for j in range(k):
            sum3 = 0
            sum4 = 0
            for i in range(len(x)):
                sum3 += ((resp[i][j]) * x[i])
            result1 = sum3 / N_j[j]
            updated_MU.append(result1)

            for i in range(len(x)):
                sum4 += (resp[i][j]) * ((x[i] - result1) ** 2)
            result2 = sum4 / N_j[j]
            if result2 < 10 ** (-20):
                result2 = 10 ** -19
            updated_var.append(result2)

            result3 = N_j[j] / N
            if result1 < 10**-10:
                result3 = 10**-5

            updated_weights.append(result3)

            sum_weight = 0
            for i in range(len(updated_weights)):
                sum_weight += updated_weights[i]



        mean = np.array(updated_MU)
        var = np.array(updated_var)
        weights = np.array(updated_weights)/sum_weight

        return [weights, mean, var]


for row_n in range(height):         #(height):
  for col_n in range(width):               #(width):
    pixel_process = []
    count_of_frames = 0
    for filename in os.listdir("D:\ELL784_Assignment_1\Grayscale_Images/"):
      if filename.endswith("jpg") and count_of_frames < 50:
        img1 = cv2.imread("D:\ELL784_Assignment_1\Grayscale_Images/frame_gray%d.jpg" %count_of_frames)
        count_of_frames += 1
        pixel_process.append(img1[row_n, col_n,0])
    print(f"pixel process for initial 50 frames for {row_n}.{col_n} pixel")

    gm = gaussian_model(pixel_process, 3)
    g_w= gm[0].tolist()
    weight_nan = np.isnan(g_w)
    weight_nan_sum = 0

    for indices in range(len(weight_nan)):
        if weight_nan[indices] == True:
            g_w[indices] = 0.0001

    for elements in g_w:
        weight_nan_sum+=elements

    gm_weightsnp= np.array(g_w) / weight_nan_sum
    g_w = gm_weightsnp.tolist()


    g_m= gm[1].tolist()
    mean_nan = np.isnan(g_m)
    for indices in range(len(mean_nan)):
        if mean_nan[indices] == True:
            g_m[indices] = 0



    g_v= gm[2].tolist()
    var_nan = np.isnan(g_v)
    for indices in range(len(var_nan)):
        if var_nan[indices] == True:
            g_v[indices] = 10**-19

    G_comp[row_n][col_n] = [g_w, g_m , g_v]


count_of_frames = 0

for filename in os.listdir("D:\ELL784_Assignment_1\Grayscale_Images/"):
    if filename.endswith("jpg"):
        img1 = cv2.imread("D:\ELL784_Assignment_1\Grayscale_Images/frame_gray%d.jpg" %count_of_frames)
        bs = BackgroundSubtraction(gaussian_count=3, gaussian_model=G_comp, alpha=0.01, t=0.5)
        bs.check_frame_wise(img1)
        G_comp = bs.gauss_model

        cv2.imwrite("D:\ELL784_Assignment_1\Foreground/fg%d.jpg" %count_of_frames, bs.img_fg)
        cv2.imwrite("D:\ELL784_Assignment_1\Background/bg%d.jpg" %count_of_frames, bs.img_bg)
        count_of_frames += 1
        print(count_of_frames)


# Set the input video path
input_video_path = r"D:\ELL784_Assignment_1\umcp.mpg"

# Create output folders if they don't exist
output_video_path_bg = r"D:\ELL784_Assignment_1\output_video_bg.avi"
output_video_path_fg = r"D:\ELL784_Assignment_1\output_video_fg.avi"
folder_path_bg = r"D:\ELL784_Assignment_1\Background"
folder_path_fg = r"D:\ELL784_Assignment_1\Foreground"

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error opening video file.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create video writer object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video_bg = cv2.VideoWriter(output_video_path_bg, fourcc, fps, (width, height))
#out_video_fg = cv2.VideoWriter(output_video_path_fg, fourcc, fps, (width, height))
count=0
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Split HSV channels
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_img_bg = hsv_img
    #hsv_img_fg = hsv_img
    gray_frame_bg = cv2.imread(r"D:\ELL784_Assignment_1\Background/bg%d.jpg" %count)
    #gray_frame_fg = cv2.imread(r"D:\ELL784_Assignment_1\Foreground/fg%d.jpg" %count)
    hsv_img_bg[:,:,2]=gray_frame_bg[:,:,0]
    #hsv_img_fg[:,:,2]=gray_frame_fg[:,:,0]
    rgb_img_bg = cv2.cvtColor(hsv_img_bg , cv2.COLOR_HSV2BGR)
    #rgb_img_fg = cv2.cvtColor(hsv_img_fg , cv2.COLOR_HSV2BGR)

    # Write the frame to the output video

    out_video_bg.write(rgb_img_bg)
    #out_video_fg.write(rgb_img_fg)
    print(count)
    count+=1
    # Break the loop if the 'Esc' key is pressed
    if cv2.waitKey(30) == 27:
        break

# Release the video capture object, video writer object, and close all windows
cap.release()
out_video_bg.release()
#out_video_fg.release()
cv2.destroyAllWindows()


