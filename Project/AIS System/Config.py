global_patient = "22"
global_case = "31"
global_slc = "1"

g_d = 7
g_std_t = 0.65
Img_shape = (256, 256)
global_all = '{}_case_{}_{}'.format(global_patient, global_case, global_slc)
path_file = r"/home/aiteam_share/database/ISLES2018_brain_aif/{}".format(global_all)

# create your fold for fitting bone and aif mask
bone_aif_mask = r"/home/dxy/ais/aif_mask/"

path_image = r"/home/aiteam_share/database/ISLES2018_4D"
path_mask = r"/home/aiteam_share/database/ISLES2018_mask"

# ---------------------Test Path--------------------------
data_root = r"/home/aiteam_share/database/ISLES2018_brain_aif/"
test_root = r"/home/dxy/ais/AIFDetectionTest"
Kmeans_Mask = "/Kmeans/AifMask"
Kmeans_Img = "/Kmeans/AifImages"
Condition_Mask = "/Conditions/AifMask"
Condition_Img = "/Conditions/AifImages"
Num_Aif = 10
Device = "cuda:0"
