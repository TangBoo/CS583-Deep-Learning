global_patient = "24"
global_case = "34"
global_slc = "1"

g_d = 7
g_std_t = 0.65
Img_shape = (256, 256)
global_all = '{}_case_{}_{}'.format(global_patient, global_case, global_slc)
path_file = r"/home/aiteam_share/database/ISLES2018_brain_aif/{}".format(global_all)

# create your fold for fitting bone and aif mask
bone_aif_mask = r"/home/dxy/ais/aif_mask/{}".format(global_all)
path_image = r"/home/aiteam_share/database/ISLES2018_4D"
path_mask = r"/home/aiteam_share/database/ISLES2018_mask"


