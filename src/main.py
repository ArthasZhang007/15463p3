from dataclasses import dataclass
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt 
import math
import scipy
import cv2

def convert(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

def readimage(path):
    return cv2.imread(path).astype(np.float32) / 255.0

def showimage(image):
    plt.imshow(convert(image))
    plt.show()

def writeimage(file, image):
    return cv2.imwrite('./data/output/' + file, (np.clip(image,0,1) * 255).astype(np.uint8))


def Gaussian(x, s):
    return np.exp(-0.5 * (x/s) * (x/s)) / (s * np.sqrt(2.0 * np.pi))

def tuplize(x):
    ans = x.view(dtype=np.dtype([('x', x.dtype), ('y', x.dtype), ('z', x.dtype)]))
    return ans.reshape(ans.shape[:-1])

def bilateral(prefix, image, flash_image, d, sigma_r, sigma_s, format):
    # reference
    
    # reference = cv2.bilateralFilter(image, d, sigma_r, sigma_s)
    # output_file = '{:s}_standard_{:s}_{:d}_{:.2f}_{:.2f}{:s}'.format(prefix,'reference', d, sigma_r, sigma_s, format)
    # writeimage(output_file, reference)
    # return

    shape = (image.shape[0], image.shape[1])
    Jc = list()
    mx = np.max(image)
    mi = np.min(image)
    NB_SEGMENTS = int(np.ceil((mx - mi)/sigma_r))

    for channel in range(3):
        J = np.zeros(shape)
        I = image[:, :, channel]
        FI = flash_image[:, :, channel]

        ind = np.indices(I.shape)
        zs = list()
        values = list()

        for j in range(NB_SEGMENTS + 1):
            i = mi + j * (mx - mi)/NB_SEGMENTS
            G = Gaussian(FI - i, sigma_r)
            K = cv2.GaussianBlur(G, (d, d), sigma_s)
            H = G * I 
            H_star = cv2.GaussianBlur(H, (d, d), sigma_s)
            dJ = H_star / K

            zs.append(i)
            values.append(dJ)
        
        x = np.arange(0, I.shape[0])
        y = np.arange(0, I.shape[1])
        z = np.array(zs)
        points = (z,x,y)


        values = np.array(values)


        input = np.dstack((FI.flatten(),ind[0].flatten(), ind[1].flatten()))



        J = scipy.interpolate.interpn(points, values, input[0])
        
        J = J.reshape(I.shape)

        Jc.append(J)



    output_file = '{:s}_join_{:s}_{:d}_{:.2f}_{:.2f}{:s}'.format(prefix,'self', d, sigma_r, sigma_s, format)
    output = cv2.merge(np.array(Jc).astype(np.float32))
    #writeimage(output_file, output)
    #showimage(image)
    return output

def linearize(C):
    return (C < 0.0404482)/12.92 + (C > 0.0404482) * np.power((C+0.055)/1.055, 2.4)    




def part1():
    ISO_A = 1600 
    ISO_F = 200
    item = 'lamp'
    typ_file = 'tif'

    A = readimage('./data/{:s}/{:s}_ambient.{:s}'.format(item, item, typ_file))
    F = readimage('./data/{:s}/{:s}_flash.{:s}'.format(item, item, typ_file))

    eps = 0.01
    tau = 0.05
    A_base = bilateral(item, A, A, 9, 0.1, 1, '.' + typ_file)
    A_NR = bilateral(item, A, F, 9, 0.1, 1, '.' + typ_file)
    F_base = bilateral(item, F, F, 9, 0.1, 1, '.' + typ_file)
    A_detail = A_NR * ((F + eps)/(F_base + eps))
    
    M = linearize(F) - linearize(A) * (ISO_F/ISO_A)

    M = (M < tau) * 1.0
    
    A_final = (1 - M) * A_detail + M * A_base

    # print(np.std(A_final - A), np.mean(A_final - A))
    # writeimage('{:s}_merge.{:s}'.format(item, typ_file), 10 * (A_final - A))
    #writeimage('{:s}_merge_diff.{:s}'.format(item, typ_file), 10 * (A_final - A))
    # writeimage('lamp_detail.tif', A_detail)
    # diff = readimage('./data/output/lamp_detail.tif') - readimage('./data/output/lamp_standard_self_9_0.10_4.00.tif')
    # plt.imshow(diff)
    # plt.show()
def boundarymask(I):
    mask = np.ones(I.shape, dtype = bool)
    mask[I.ndim * (slice(1, -1),)] = False
    return mask * 1



def grad(I, bonus=0):
    X_i = np.zeros(I.shape)
    Y_i = np.zeros(I.shape)
    X = np.diff(I, axis = 0)
    Y = np.diff(I, axis = 1)
    i = np.indices(X.shape)
    j = np.indices(Y.shape)

    X_i[i[0]+bonus,i[1]] = X[i[0],i[1]]
    Y_i[j[0],j[1]+bonus] = Y[j[0],j[1]]

    return (X_i,Y_i)

def div(I):
    u,v = I
    X_u,_ = grad(u,1)
    _, Y_v = grad(v,1)
    return X_u + Y_v

def lap(I):
    lapo = np.array(
    [
        [0,1,0],
        [1,-4,1],
        [0,1,0]
    ])
    return scipy.signal.convolve2d(I, lapo, mode = 'same')

def rescale(I):
    return (I-np.min(I))/(np.max(I)-np.min(I))



def dot(a,b):
    return np.sum(a * b)


def CGD(D, I, I_boundary, eps, N):
    B = 1 - boundarymask(I)
    I_new = B * I + (1 - B) * I_boundary
    r = B * (D - lap(I_new))
    d = r 
    delta_new = dot(r,r)
    n = 0

    while dot(r,r) > eps * eps and n < N:
        q = lap(d)
        f = delta_new/dot(d,q)
        I_new = I_new + B * (f * d)
        r = B * (r - f * q)
        delta_old = delta_new
        delta_new = dot(r,r)
        beta = delta_new/delta_old
        d = r + beta * d 
        n = n + 1 
        # print(np.std(r))
    return I_new

def part2():
    item = 'museum'
    typ_file = 'png'
    image = readimage('./data/{:s}/{:s}_ambient.{:s}'.format(item, item, typ_file))
    flash = readimage('./data/{:s}/{:s}_flash.{:s}'.format(item, item, typ_file))
    #np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)}, threshold=sys.maxsize)
    final = list()

    dAlist = list()
    dPlist = list()
    dClist = list()
    for c in range(3):
        A = image[:,:,c]
        P = flash[:,:,c]
        
        dA = grad(A)
        dP = grad(P)
        
        
        M_t = np.abs(dot(dA[0], dP[0]) + dot(dA[1],dP[1]))
        M_b = np.sqrt(dot(dA[0], dA[0]) + dot(dA[1], dA[1])) * np.sqrt(dot(dP[0], dP[0]) + dot(dP[1], dP[1]))
        M = M_t / M_b
        sigma = 20
        tau = 0.5
        dAlist.append(dA)
        dPlist.append(dP)

        w = rescale(np.tanh(sigma * (rescale(P) - tau)))

        I_x = w * dA[0] + (1-w)*(M*dP[0] + (1-M)*dA[0])
        I_y = w * dA[1] + (1-w)*(M*dP[1] + (1-M)*dA[1])

        dClist.append((I_x, I_y))
        I = CGD(div((I_x,I_y)), np.zeros(A.shape), A, 0.01, 1000)
        final.append(I)
        #print(np.std(I_new - I))
        # assert(np.std(I_new - I) < eps)
    final = np.dstack((final[0], final[1], final[2]))
    writeimage("{:s}_final_20_0.5.{:s}".format(item, typ_file), final)

    # def split(lis):
    #     a = list()
    #     b = list()
    #     for (x,y) in lis:
    #         a.append(x)
    #         b.append(y)
    #     return (a,b)
    # def combine(list):
    #     return np.dstack((list[0], list[1], list[2]))
    # writeimage("dA_x.png", rescale(combine(split(dAlist)[0])))
    # writeimage("dA_y.png", rescale(combine(split(dAlist)[1])))
    # writeimage("dP_x.png", rescale(combine(split(dPlist)[0])))
    # writeimage("dP_y.png", rescale(combine(split(dPlist)[1])))
    # writeimage("dC_x.png", rescale(combine(split(dClist)[0])))
    # writeimage("dC_y.png", rescale(combine(split(dClist)[1])))

    #plt.imshow(rescale(combine(split(dAlist)[0])))
    #plt.show()

def main():
    I = readimage('data/lamp/lamp_ambient.tif')
    standard = (readimage('data/output/lamp_standard_self_9_0.10_4.00.tif'))
    join = (readimage('data/output/lamp_join.tif'))
    detail = (readimage('data/output/lamp_detail.tif'))
    merge = (readimage('data/output/lamp_final.tif'))
    print(writeimage('lamp_standard.jpg', standard))
    print(writeimage('lamp_join.jpg', join))
    print(writeimage('lamp_detail.jpg', detail))
    print(writeimage('lamp_merge.jpg', merge))

    print(writeimage('lamp_standard_diff.jpg', 10 * np.abs(standard - I)))
    print(writeimage('lamp_join_diff.jpg', 10 * np.abs(join - standard)))
    print(writeimage('lamp_detail_diff.jpg', 10 * np.abs(detail - join)))
    print(writeimage('lamp_merge_diff.jpg', 10 * np.abs(merge - detail)))
    #part1()
    #part2()

main()