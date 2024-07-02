import numpy as np
import math


class BackgroundSubtraction:
    def __init__(self, gaussian_count, gaussian_model, alpha, t):
        self.n = gaussian_count
        self.gauss_model = gaussian_model
        self.alpha = alpha
        self.t = t

    def normal_distribution(self, x, mean, variance):
        power_term = ((x - mean) ** 2 / variance) / 2
        numerator = math.exp((-1) * power_term)
        denominator = ((variance * 2) * math.pi) ** 0.5
        value = numerator / denominator
        return value

    # gm_pixel: gaussian model parameters for pixel
    # def priority(self, gm_pixel):
    #     order = gm_pixel[0] / np.sqrt(gm_pixel[2])
    #     idx = np.argsort(order)[::-1]
    #     gm_mean = gm_pixel[1][idx]
    #     gm_var = gm_pixel[2][idx]
    #     gm_weight = gm_pixel[0][idx]
    #     gm_pixel_updated = [gm_weight, gm_mean, gm_var]
    #     return gm_pixel_updated

######### PRIORITY ##########################
    def priority (self, gm_pixel):
        order = []
        for index in range(self.n):
            prior=gm_pixel[0][index]/np.sqrt(gm_pixel[2][index])
            order.append(prior)
        idx = np.argsort(order)[::-1]

        G_weight = []
        G_mean = []
        G_var = []
        prior_index=0
        for new in idx:
            G_weight.append(gm_pixel[0][new])
            G_mean.append(gm_pixel[1][new])
            G_var.append(gm_pixel[2][new])

        gm_pixel_updated = [G_weight, G_mean, G_var]
        return gm_pixel_updated


    def condition_check(self, gaussian_number, gm_pixel):
        weight_sum = 0
        index = 0

        while weight_sum < self.t:
            weight_sum += gm_pixel[0][index]
            index += 1

        B = index

        if gaussian_number < B:
            bg_flag = True
        else:
            bg_flag = False

        return bg_flag

    def update_gaussian(self, gm_pixel, gaussian_number, pixel_value):

        if gaussian_number < self.n:
            rho = self.alpha * self.normal_distribution(pixel_value, gm_pixel[1][gaussian_number], gm_pixel[2][gaussian_number])
            M = 0
            for indices in range(self.n):
                if indices == gaussian_number:
                    M = 1

                else:
                    M = 0
                gm_pixel[0][indices]=(1 - self.alpha)*gm_pixel[0][indices] + self.alpha*M
            #gm_pixel[0][gaussian_number] = (1 - self.alpha) * gm_pixel[0][gaussian_number] + self.alpha
            gm_pixel[1][gaussian_number] = (1 - rho) * gm_pixel[1][gaussian_number] + (rho * pixel_value)
            gm_pixel[2][gaussian_number] = (1 - rho) * gm_pixel[2][gaussian_number] + (rho * ((pixel_value - gm_pixel[1][gaussian_number]) ** 2))
            #gm_pixel[2][gaussian_number] = np.abs(gm_pixel[2][gaussian_number])

        else:
            gm_pixel[1][self.n - 1] = pixel_value
            gm_pixel[2][self.n - 1] = 225
            gm_pixel[0][self.n - 1] = 0.175

        # sum = 0
        # for l in range(self.n):
        #     sum += gm_pixel[0][l]

        # for l in range(self.n):
        #     gm_pixel[0][l] = gm_pixel[0][l] / sum

        return gm_pixel

    def check_frame_wise(self, img):
        row = len(self.gauss_model)
        column = len(self.gauss_model[0])
        self.img_fg = np.zeros((row, column), dtype="uint8")
        self.img_bg = np.zeros((row, column), dtype="uint8")
        for i in range(row):
            for j in range(column):
                gm_pixel = self.gauss_model[i][j]
                gm_pixel_updated = self.priority(gm_pixel)
                pixel_value = img[i, j, 0]
                flag = False
                diff = []
                for k in range(self.n):

                    difference = pixel_value - (gm_pixel_updated[1][k])
                    if np.abs(difference) < (2.5 * np.sqrt(gm_pixel_updated[2][k])):
                        # print(f"present in gaussian {k}")
                        # g_n = k
                        flag = True
                        difference2 = np.abs(difference) - (2.5 * np.sqrt(gm_pixel_updated[2][k]))
                        diff.append(np.abs(difference2))
                        # break

                    else:
                        diff.append(1000000000)

                if flag is False:
                    gaussian_number = self.n + 1
                else:
                    gaussian_number = diff.index(min(diff))
                    # print("Not present in any gaussian")
                cond_flag = self.condition_check(gaussian_number, gm_pixel_updated)
                # print(cond_flag)
                if cond_flag:
                    self.img_bg[i, j] = pixel_value
                    self.img_fg[i, j] = 255

                else:
                    self.img_bg[i, j] = gm_pixel_updated[1][0]
                    self.img_fg[i, j] = pixel_value

                self.gauss_model[i][j] = self.update_gaussian(gm_pixel_updated, gaussian_number, pixel_value)
