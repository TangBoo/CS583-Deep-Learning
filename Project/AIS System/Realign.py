import numpy as np
import matplotlib.pyplot as plt
import cv2
import itk
from itkwidgets import view
import SimpleITK as sitk
from IPython.display import clear_output
from tqdm import tqdm
from glob import glob


def getCaselst(root):
    patient_lst = glob(root + '/*/')
    case_lst = []
    for i in range(len(patient_lst)):
        temp_lst = glob(patient_lst[i] + '/*/')
        for j in range(len(temp_lst)):
            if temp_lst[j].endswith('png/'):
                continue
            case_lst.append(temp_lst[j])
    return case_lst


def load4DImage(idx, case_lst, channel=0):
    slc_lst = glob(case_lst[idx] + '/*/')
    slc_lst = sorted(slc_lst, key=lambda x: x.split('/')[-2].split('_')[0])
    # [d, time, h, w]
    anat_imgs = []
    for i in range(len(slc_lst)):
        time_imgs = []
        time_lst = glob(slc_lst[i] + '/*')
        time_lst = sorted(time_lst, key=lambda x: int(x.split('/')[-1].split('.')[0].split('img')[-1]))
        for j in range(len(time_lst)):
            time_imgs.append(cv2.imread(time_lst[j], channel))
        anat_imgs.append(time_imgs)
    return np.array(anat_imgs).swapaxes(0, 1)


def load4DDcm(idx, case_lst):
    slc_lst = glob(case_lst[idx] + '/*/')
    slc_lst = sorted(slc_lst, key=lambda x: x.split('/')[-2].split('_')[0])
    anat_time_imgs = []
    for i in range(len(slc_lst)):
        time_imgs = []
        time_lst = glob(slc_lst[i] + '/*')
        time_lst = sorted(time_lst, key=lambda x: int(x.split('/')[-1].split('.')[0].split('img')[-1]))
        for j in range(len(time_lst)):
            time_imgs.append(sitk.ReadImage(time_lst[j]))
        anat_time_imgs.append(time_imgs)
    time_lst = [len(anat_time_imgs[i]) for i in range(len(anat_time_imgs))]
    time = min(time_lst)
    time_anat_imgs = [[] for _ in range(time)]
    for i in range(time):
        for j in range(len(anat_time_imgs)):
            time_anat_imgs[i].append(anat_time_imgs[j][i])
    return time_anat_imgs


def getSpace(idx, case_lst):
    slc_lst = glob(case_lst[idx] + '/*/')
    time_lst = glob(slc_lst[0] + '/*')
    img_sitk = sitk.ReadImage(time_lst[0])
    return img_sitk.GetSpacing()


def seriesImg_show(imgs, time=0.8):
    '''
    :param imgs: [time or d, h, w]
    :param time: pass
    :return: none
    '''
    for i in range(len(imgs)):
        plt.figure(figsize=(15, 7))
        clear_output(wait=True)
        plt.imshow(imgs[i])
        plt.show()
        plt.pause(time)
        plt.close()


def Viewer_3D(imgs):
    '''
    :param imgs: array:[D, H, W]
    :return: the 3D player
    '''
    itk_img = itk.image_view_from_array(imgs)
    viewer = view(itk_img, rotate=True, axes=True, vmin=0, vmax=255, gradient_opacity=0.9)
    return viewer


def getMaxArea(img):
    dist = cv2.distanceTransform(src=img.astype('uint8'), distanceType=cv2.DIST_L2, maskSize=5)
    mask = dist.copy()
    ret1, the = cv2.threshold(dist.astype('uint8'), dist.min(), dist.max(), cv2.THRESH_OTSU)
    dist[dist > ret1] = 0
    ret2, _ = cv2.threshold(dist.astype('uint8'), dist.min(), dist.max(), cv2.THRESH_OTSU)
    mask[mask < ret2] = 0
    mask = mask.astype('bool').astype('uint8')
    ignore_location = np.where(mask == 0)
    return mask, ignore_location


def ellipse_fitting(contour):
    edge = cv2.Canny(contour.astype('uint8'), 0, 1)
    y, x = np.nonzero(edge)
    edge_list = np.array([[x_, y_] for x_, y_ in zip(x, y)])
    _ellipse = cv2.fitEllipse(edge_list)
    edge_clone = edge.copy()
    cv2.ellipse(edge_clone, _ellipse, (255, 0, 0), 2)
    edge_clone[edge != 0] = 0
    return edge_clone, _ellipse


def mutualInfo_rigister_3D(fixed, moving, space=(1, 1, 1)):
    fixed = sitk.GetImageFromArray(fixed)
    fixed.SetSpacing(space)
    moving = sitk.GetImageFromArray(moving)
    moving.SetSpacing(space)
    initial_transform = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    moving_resampled = sitk.Resample(moving, fixed, initial_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    # registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    # registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    # registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    optimized_transform = sitk.Euler3DTransform()
    registration_method.SetMovingInitialTransform(initial_transform)
    registration_method.SetInitialTransform(optimized_transform, inPlace=False)
    final_transform = registration_method.Execute(sitk.Cast(fixed, sitk.sitkFloat32),
                                                  sitk.Cast(moving, sitk.sitkFloat32))
    ans = sitk.Resample(moving, referenceImage=fixed, interpolator=sitk.sitkLinear, transform=final_transform,
                        useNearestNeighborExtrapolator=True)
    ans = sitk.GetArrayFromImage(ans)
    return ans


def rigid_registration_3D(imgs, template_shift=(0, 0, 0), template_rotation=(0, 0, 0), isRotate=True,
                          space=(1, 1, 1)):
    slice_num = imgs.shape[0]
    ori_imgs = imgs.copy()
    imgs = imgs.copy()
    d, h, w = imgs.shape
    Slia = np.max(imgs, axis=0)
    Slia, ignore_loc = getMaxArea(Slia)
    for i in range(len(imgs)):
        imgs[i][ignore_loc] = 0

    Sagittal = np.max(imgs, axis=1).astype('bool')
    Coroa = np.max(imgs, axis=2).astype('bool')

    _, Sila_info = ellipse_fitting(Slia)
    _, Sagi_info = ellipse_fitting(Sagittal)
    _, Coroa_info = ellipse_fitting(Coroa)

    (x, y) = Sila_info[0]
    z = Coroa_info[0][1]
    theta_x, theta_y, theta_z = Sagi_info[-1], Coroa_info[-1], Sila_info[-1]

    if not isRotate:
        rotate_out = (theta_x, theta_y, theta_z)
        shift_out = (x, y, z)
        theta_x, theta_y, theta_z = 0.0, 0.0, 0.0
    else:
        theta_x = theta_x - template_rotation[0]
        theta_y = theta_y - template_rotation[1]
        theta_z = theta_z - template_rotation[2]
        shift_out = (x - template_shift[0], y - template_shift[1], z - template_shift[2])
        rotate_out = (theta_x, theta_y, theta_z)

    offset = (-space[0] * (w // 2 - x), -space[1] * (h // 2 - y), 0)  # (x, y, z)
    # roatation axis: r_x, r_y, r_z, roatation center: z, y, x
    # translation = sitk.TranslationTransform(3, offset)
    translation = sitk.Euler3DTransform((w//2, h//2, slice_num//2), 0, 0, np.radians(theta_z), offset)
    imgs_sitk = sitk.GetImageFromArray(ori_imgs)  # [z, y, x]->[y, x, z]
    imgs_sitk.SetSpacing(space)
    out = sitk.Resample(imgs_sitk, transform=translation)
    shift_out = (x - template_shift[0], y - template_shift[1], z - template_shift[2])
    return sitk.GetArrayFromImage(out), shift_out, rotate_out


def Registration(anat_time_img, space=(1.0, 1.0, 1.0)):
    time, d, h, w = anat_time_img.shape
    template_img, template_shift, template_rotate = rigid_registration_3D(anat_time_img[0], template_shift=(0, 0, 0),
                                                                          template_rotation=(0, 0, 0), isRotate=False,
                                                                          space=space)

    output_shift = {'x': [0, ], 'y': [0, ], 'z': [0, ]}
    output_rotation = {'x': [0, ], 'y': [0, ], 'z': [0, ]}
    output = [template_img]
    for i in tqdm(range(1, time)):
        temp_img, temp_shift, temp_rotate = rigid_registration_3D(anat_time_img[i], template_shift=template_shift,
                                                                  template_rotation=template_rotate, space=space,
                                                                  isRotate=True)
        temp_img = mutualInfo_rigister_3D(template_img, temp_img, space)
        output_shift['x'].append(temp_shift[0])
        output_shift['y'].append(temp_shift[1])
        output_shift['z'].append(temp_shift[2])
        output_rotation['x'].append(temp_rotate[0])
        output_rotation['y'].append(temp_rotate[1])
        output_rotation['z'].append(temp_rotate[2])
        template_img = temp_img
        output.append(temp_img)
    output = np.array(output)
    return output, output_shift, output_rotation


def plot_moveInfor(shifts, rotations):
    x_shift = np.array(shifts['x'])
    y_shift = np.array(shifts['y'])
    z_shift = np.array(shifts['z'])
    x_rotate = np.array(rotations['x'])
    y_rotate = np.array(rotations['y'])
    z_rotate = np.array(rotations['z'])
    x_shift[1:] -= x_shift.mean()
    y_shift[1:] -= y_shift.mean()
    z_shift[1:] -= z_shift.mean()
    x_rotate[1:] -= x_rotate.mean()
    y_rotate[1:] -= y_rotate.mean()
    z_rotate[1:] -= z_rotate.mean()
    plt.figure(figsize=(15, 7))
    plt.subplot(3, 2, 1)
    plt.plot(x_shift)
    plt.axis([0, 30, -10, 10])
    plt.title('x shift')
    plt.subplot(3, 2, 2)
    plt.plot(x_rotate)
    plt.axis([0, 30, -2, 2])
    plt.title('x rotation')
    plt.subplot(3, 2, 3)
    plt.plot(y_shift)
    plt.axis([0, 30, -10, 10])
    plt.title('y shift')
    plt.subplot(3, 2, 4)
    plt.plot(y_rotate)
    plt.axis([0, 30, -2, 2])
    plt.title('y rotation')
    plt.subplot(3, 2, 5)
    plt.plot(z_shift)
    plt.axis([0, 30, -10, 10])
    plt.title('z shift')
    plt.subplot(3, 2, 6)
    plt.plot(z_rotate)
    plt.axis([0, 30, -2, 2])
    plt.title('z rotation')
    plt.show()


#
if __name__ == "__main__":
    png_path = "/data/aiteam_ctp/database/AIS_210713/0713_dst_png"
    dcm_path = "/data/aiteam_ctp/database/AIS_210713/0713_dst"

    case_list_dcm = getCaselst(dcm_path)
    case_list_png = getCaselst(png_path)

    # time_anat_imgs_dcm = load4DDcm(0, case_list_dcm)
    idx = -1
    time_anat_imgs = load4DImage(idx, case_list_png, 0)
    space = getSpace(idx, case_list_dcm)
    output, shifts, rotations = Registration(time_anat_imgs, space=space)

    seriesImg_show(output[:, 0, ...], 0.2)
    plot_moveInfor(shifts, rotations)