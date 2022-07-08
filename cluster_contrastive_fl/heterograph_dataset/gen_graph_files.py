import csv
import numpy as np 
import pandas as pd 

# ift2ift = pd.read_csv('ifttt_interact_ifttt.csv')
# ift2smt = pd.read_csv('./cross_inter/ifttt2smartthings.csv')
# ift_in_ift2smt = ift2smt['ifttt'].to_list()
# graph_groups =  ift2ift.groupby('graph_id')

# for graph_id in graph_groups.groups:
# 	heter_graph = graph_groups.get_group(graph_id)
# 	src = heter_graph['src'].to_numpy()
# 	dst = heter_graph['dst'].to_numpy()
# 	with open('./ifttt_interact_smartthings.csv','a+', encoding='utf-8', newline='') as f:
# 		for idx in src:
# 			if idx in ift_in_ift2smt:
# 				kk = ift_in_ift2smt.index(idx)
# 				csv.writer(f, dialect="excel").writerow((graph_id, ift2smt['ifttt'][kk], ift2smt['smt'][kk]))
# 		for idx in dst:
# 			if idx in ift_in_ift2smt:
# 				kk = ift_in_ift2smt.index(idx)
# 				csv.writer(f, dialect="excel").writerow((graph_id, ift2smt['ifttt'][kk], ift2smt['smt'][kk]))

# ift2ift = pd.read_csv('ifttt_interact_ifttt.csv')
# alx2ift = pd.read_csv('./cross_inter/alexa2ifttt.csv')
# ift_in_alx2ift = alx2ift['ifttt'].to_list()
# graph_groups =  ift2ift.groupby('graph_id')

# for graph_id in graph_groups.groups:
# 	heter_graph = graph_groups.get_group(graph_id)
# 	src = heter_graph['src'].to_numpy()
# 	dst = heter_graph['dst'].to_numpy()
# 	with open('./alexa_interact_ifttt.csv','a+', encoding='utf-8', newline='') as f:
# 		for idx in src:
# 			if idx in ift_in_alx2ift:
# 				kk = ift_in_alx2ift.index(idx)
# 				csv.writer(f, dialect="excel").writerow((graph_id, alx2ift['alexa'][kk], alx2ift['ifttt'][kk]))
# 		for idx in dst:
# 			if idx in ift_in_alx2ift:
# 				kk = ift_in_alx2ift.index(idx)
# 				csv.writer(f, dialect="excel").writerow((graph_id, alx2ift['alexa'][kk], alx2ift['ifttt'][kk]))

# smt2smt = pd.read_csv('smt_interact_smt.csv')
# ift2smt = pd.read_csv('./cross_inter/ifttt2smartthings.csv')
# smt_in_ift2smt = ift2smt['smt'].to_list()
# graph_groups =  smt2smt.groupby('graph_id')

# for graph_id in graph_groups.groups:
# 	heter_graph = graph_groups.get_group(graph_id)
# 	src = heter_graph['src'].to_numpy()
# 	dst = heter_graph['dst'].to_numpy()
# 	with open('./ifttt_interact_smartthings.csv','a+', encoding='utf-8', newline='') as f:
# 		for idx in src:
# 			if idx in smt_in_ift2smt:
# 				kk = smt_in_ift2smt.index(idx)
# 				csv.writer(f, dialect="excel").writerow((graph_id, ift2smt['ifttt'][kk], ift2smt['smt'][kk]))
# 		for idx in dst:
# 			if idx in smt_in_ift2smt:
# 				kk = smt_in_ift2smt.index(idx)
# 				csv.writer(f, dialect="excel").writerow((graph_id, ift2smt['ifttt'][kk], ift2smt['smt'][kk]))

# smt2smt = pd.read_csv('smt_interact_smt.csv')
# ift2smt = pd.read_csv('./cross_inter/alexa2smartthings.csv')
# smt_in_ift2smt = ift2smt['smt'].to_list()
# graph_groups =  smt2smt.groupby('graph_id')

# for graph_id in graph_groups.groups:
# 	heter_graph = graph_groups.get_group(graph_id)
# 	src = heter_graph['src'].to_numpy()
# 	dst = heter_graph['dst'].to_numpy()
# 	with open('./alexa_interact_smartthings.csv','a+', encoding='utf-8', newline='') as f:
# 		for idx in src:
# 			if idx in smt_in_ift2smt:
# 				kk = smt_in_ift2smt.index(idx)
# 				csv.writer(f, dialect="excel").writerow((graph_id, ift2smt['alexa'][kk], ift2smt['smt'][kk]))
# 		for idx in dst:
# 			if idx in smt_in_ift2smt:
# 				kk = smt_in_ift2smt.index(idx)
# 				csv.writer(f, dialect="excel").writerow((graph_id, ift2smt['alexa'][kk], ift2smt['smt'][kk]))