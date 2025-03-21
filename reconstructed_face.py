import os

import pymeshlab
import numpy as np
#face
#E:\BaiduNetdiskDownload\STLmesh\face_bone_data
from matplotlib import pyplot as plt

path = r'F:\BaiduNetdiskDownload\result\result\ours20480\dataview'
pathlist = os.listdir(path)
save_path = r'F:\zrs\project\pythonProject\get_hd\face'
import glob

hd_mean = []
rms = []
hd_min = []
hd_max = []

for i in pathlist:
    if(i.__contains__('face_predicted')):
        name = i.strip('face_predicted.txt')
        if(int(name) < 10):
            loadmesh = glob.glob('E:/BaiduNetdiskDownload/STLmesh/face_bone_data/0' + name + '/soft/*.stl')
        else:
            loadmesh = glob.glob('E:/BaiduNetdiskDownload/STLmesh/face_bone_data/' + name + '/soft/*.stl')


        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(loadmesh[0])
        ms.save_current_mesh(save_path + '/ground_mesh/' + name + ".ply")

        point_cloud = np.loadtxt(path + '/' + i)

        # 2. 创建 MeshSet
        ms = pymeshlab.MeshSet()

        # 3. 用点云创建一个空网格
        mesh = pymeshlab.Mesh(vertex_matrix=point_cloud)

        # 4. 添加网格到 MeshSet
        ms.add_mesh(mesh, "pointcloud")

        # 创建 MeshSet 对象

        # 1. 导入点云（支持 .ply, .obj, .xyz 等格式）
        # ms.save_current_mesh("cleaned_pointcloud.ply")
        ms.compute_normal_for_point_clouds(k=100, smoothiter=0, flipflag=False)


        ms.apply_normal_point_cloud_smoothing(k=30)

        #pymeshlab.print_filter_list()

        ms.generate_surface_reconstruction_screened_poisson(depth=10, samplespernode=2)
        ms.meshing_re_orient_faces_coherently()


        ms.compute_selection_by_condition_per_vertex(condselect='(q < 5.8)')
        ms.meshing_remove_selected_vertices()
        # ms.save_current_mesh("cleaned_pointcloud.ply")

        ms.meshing_surface_subdivision_loop()



        ms.apply_coord_taubin_smoothing(stepsmoothnum=100)
        ms.meshing_close_holes()


        ms.save_current_mesh(save_path + '/pred_mesh/' + name + ".ply")



        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(loadmesh[0])
        num = ms.current_mesh().vertex_number()
        ms.load_new_mesh(save_path + '/pred_mesh/' + name + ".ply")

        hd = ms.get_hausdorff_distance(samplenum=num)
        ms.compute_color_from_scalar_per_vertex(minval = 0, maxval = 10,colormap="RGB")
        #ms.compute_color_from_scalar_per_vertex(minval=hd['min'], maxval=hd['max'], colormap="RGB")
        ms.save_current_mesh(save_path + '/hd/' + name + ".ply")


        hd_min.append(hd['min'])
        hd_mean.append(hd['mean'])
        hd_max.append(hd['max'])
        rms.append(hd['RMS'])
        print(hd)
        # 获取 Hausdorff 误差直方图
        bin_ranges = ms.get_scalar_histogram_per_vertex()



        # 计算每个柱子的中心点
        bin_centers = (bin_ranges['hist_bin_min'][:-1] + bin_ranges['hist_bin_max'][:-1]) / 2

        # 绘制柱状图
        plt.figure(figsize=(8, 5))
        plt.bar(bin_centers, bin_ranges['hist_count'][:-1], width=np.diff(bin_ranges['hist_bin_min']), edgecolor='black', alpha=0.7, color='blue')

        plt.xlabel("Hausdorff Distance")
        plt.ylabel("Vertex Count")
        plt.title("Hausdorff Distance Distribution")
        plt.grid(True)
        #plt.show()
        plt.savefig(save_path + '/hd_fig/' + name + ".jpg")

print(np.mean(rms))
print(np.std(rms))
print(np.mean(hd_min))
print(np.std(hd_min))
print(np.mean(hd_mean))
print(np.std(hd_mean))
print(np.mean(hd_max))
print(np.std(hd_max))