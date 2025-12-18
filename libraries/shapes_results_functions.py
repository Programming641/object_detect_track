
from libraries import pixel_shapes_functions, pixel_functions, image_functions, same_shapes_functions, btwn_amng_files_functions
from libraries.cv_globals import top_shapes_dir, top_images_dir
from libraries import cv_globals

from PIL import Image
import math
import os, sys
import copy, pickle        


# I want to get all matches from all_consecutives that are sitting on the pixels I provide.
# all_consecutives have to have this data format
# [ { each files: [ ( image1 pixels, matched image2 pixels, im1to2 movement, ... ), ... ], ... }, ... ]
# 
# consecs_lookups_by_p
# Lookup will be like this.
# every pixel will have its ( consecutive index, match number ).
#
# { each files: [  { pixel: { ( consec index, match number ), ... }, ... }, { pixel:  { ( consec index, match number ), ... }, ... } ], ... }
# usage example. consecs_lookups_by_p[each_files][image_number][pixel] to get the consecutive match that this pixel is sitting on.
# specifically getting this: { (consec_index, m_index ), ... }
def prepare_consecs_lookups( all_each_files, all_consecutives ):

	consecs_lookups_by_p = {} # { each files: [ image1 consecutive by pixel, image2 consecutive by pixel ], ... }
							  # image consecutive by pixel:  { pixel: { ( consecutive index, match index ), ... }, ... }
	for eachf in all_each_files:
		consecs_lookups_by_p[eachf] = [ {}, {} ]

	for consec_index, consec in enumerate(all_consecutives):
		
		for eachf in consec:
			for m_index, match in enumerate( consec[eachf] ):

				for im1or2 in range(2):
					for pixel in match[im1or2]:
						if consecs_lookups_by_p[eachf][im1or2].get(pixel) is None:
							consecs_lookups_by_p[eachf][im1or2][pixel] = { ( consec_index, m_index ) }
						else:
							consecs_lookups_by_p[eachf][im1or2][pixel].add( (consec_index, m_index ) )

	return consecs_lookups_by_p



# add_list: [ ( consecutive index, each files, m_index, match_data ), ... ]
# match_data: ( image1 pixels, image2 pixels, movement )
def add_to_consecs_lookups( add_list, consecs_lookups ):

	for add_data in add_list:
		# ( consecutive index, each files, m_index, ( image1 pixels, image2 pixels, im1to2 movement ) )
		
		consec_index = add_data[0]
		each_files = add_data[1]
		m_index = add_data[2]
		match = add_data[3]
		
		if not consecs_lookups.get(each_files):
			consecs_lookups[each_files] = [ {}, {} ]
		
		for im1or2 in range(2):
			for pixel in match[im1or2]:
				if consecs_lookups[each_files][im1or2].get(pixel):
					consecs_lookups[each_files][im1or2][pixel].add( (consec_index, m_index) )
				else:
					consecs_lookups[each_files][im1or2][pixel] = { (consec_index, m_index ) }



# all_m_lookups_by_p
# ・ match index lookup in all matches. so given the pixel to get the match index in all matches.
# { each files: [ image1 match indexes by pixel, image2 match indexes by pixel ], ... }
# image match indexes by pixel:  { pixel: { match indexes }, ... }
# 
# all_unm_im_pixels
# { each files: [ image1 unmatched pixels, image2 unmatched pixels ], ... }
def get_all_m_lookups_n_all_unm_p( all_each_files_matches, image_size ):

	all_m_lookups_by_p = {}

	all_unm_im_pixels = {} # { each files: [ image1 unmatched pixels, image2 unmatched pixels ], ... }
	
	for eachf in all_each_files_matches:
		all_m_lookups_by_p[eachf] = [ {}, {} ]

		still_unm_im_pixels = [ set(), set() ]
		for x in range(0, image_size[0] ):
			for y in range(0, image_size[1] ):
				still_unm_im_pixels[0].add( (x,y) )
				still_unm_im_pixels[1].add( (x,y) )

		for m_index, each_m in enumerate( all_each_files_matches[eachf] ):
			for im1or2 in range(2):
				
				for pixel in each_m[im1or2]:
			
					if all_m_lookups_by_p[eachf][im1or2].get(pixel) is None:
						all_m_lookups_by_p[eachf][im1or2][pixel] = { m_index }
					else:
						all_m_lookups_by_p[eachf][im1or2][pixel].add(m_index)

				still_unm_im_pixels[im1or2] = still_unm_im_pixels[im1or2].difference( each_m[im1or2] )

		all_unm_im_pixels[eachf] = still_unm_im_pixels


	return all_m_lookups_by_p, all_unm_im_pixels


			



# all_matches are the all matches that are above the threshold. this is the return value from same_shapes_functions.match_by_mov_pixels_against_im3.
# relevant documentation is in tests/new/similar_shapes/無題.png
# all_matches:  [ (matched_percent, matched_pixels, movement, out_im_pixels ), ... ]
# return: set of real match indexes.
def get_real_matches_from_all_m( all_matches ):

	def _get_m_indexes_w_nbr_mv( cur_m_index, cur_m_indexes_w_nbr_mv, all_nbr_mvs ):
		
		for m_index_w_nbr_mv in all_nbr_mvs[cur_m_index]:
			
			if m_index_w_nbr_mv in cur_m_indexes_w_nbr_mv:
				continue
			else:
				cur_m_indexes_w_nbr_mv.add(m_index_w_nbr_mv)
				_get_m_indexes_w_nbr_mv( m_index_w_nbr_mv, cur_m_indexes_w_nbr_mv, all_nbr_mvs )
			

	def _group_by_mv( all_nbr_mvs ):
		all_matches_by_mv = []
		
		done_m_indexes = set()
		for m_index in all_nbr_mvs:
			if m_index in done_m_indexes:
				continue
			
			cur_m_indexes_w_nbr_mv = {m_index}
			_get_m_indexes_w_nbr_mv( m_index, cur_m_indexes_w_nbr_mv, all_nbr_mvs )
			
			all_matches_by_mv.append(cur_m_indexes_w_nbr_mv)
			
			done_m_indexes |= cur_m_indexes_w_nbr_mv
		
		return all_matches_by_mv
	
	
	def _get_m_largest_mpixels( m_indexes ):
	
		largest_m_index = None
		largest_m_pixels_len = 0
		
		for m_index in m_indexes:
			if len( all_matches[m_index][1] ) > largest_m_pixels_len:
				largest_m_index = m_index
				largest_m_pixels_len = len( all_matches[m_index][1] )
		
		return largest_m_index


	# get neighbor movement match indexes. example. let's say, given movement is (3,5). max_diff=1. then neighbor movements are (4,5), (2,5), (3,6), (3,4)... and so on
	def _get_nbr_mv_matches( cur_m_index, m_indexes, max_diff=2 ):
		nbr_mv_m_indexes = set()

		cur_mv = all_matches[cur_m_index][2]

		x0, y0 = cur_mv
		allowed_mv_diffs = set()
		
		# Explore around the origin
		for dx in range(-max_diff, max_diff + 1):
			for dy in range(-max_diff, max_diff + 1):
				if abs(dx) + abs(dy) <= max_diff:  # Manhattan distance constraint
					if dx != 0 or dy != 0:  # exclude the original point
						allowed_mv_diffs.add((x0 + dx, y0 + dy))

		for m_index in m_indexes:
			if m_index == cur_m_index:
				continue
			
			other_m_mv = all_matches[m_index][2]
			if other_m_mv in allowed_mv_diffs:
				nbr_mv_m_indexes.add(m_index)
		
		return nbr_mv_m_indexes





	def _get_real_matches( all_matches_in_sep_mv, cur_m_index, real_matches, done_m_indexes=None, states=None ):
		
		#print("cur_m_index " + str(cur_m_index) )
		cur_m_pixels_len = len( all_matches[cur_m_index][1] )

		if done_m_indexes is None:
			# initialize
			done_m_indexes = {cur_m_index}
			states = { cur_m_index: { "going_back_up": 0, "going_back_down": 0, "cur_lowest": cur_m_pixels_len, "cur_highest": None, "cur_highest_m": None } }
			states["current_m_index"] = copy.deepcopy( states[cur_m_index] )
					
		nbr_mv_m_indexes = _get_nbr_mv_matches( cur_m_index, all_matches_in_sep_mv ) # starting with the match that has the largest matched pixels
		nbr_mv_m_indexes = nbr_mv_m_indexes.difference(done_m_indexes)
		
		last_nbr_mv_m_index = None
		returned_nbr_mv_m = None
		for nbr_mv_m_index in nbr_mv_m_indexes:
			if nbr_mv_m_index in done_m_indexes:
				continue
			done_m_indexes.add(nbr_mv_m_index)
			last_nbr_mv_m_index = nbr_mv_m_index
			
			#print("nbr_mv_m_index " + str(nbr_mv_m_index) + " called by the cur_m_index " + str(cur_m_index) + " returned_nbr_mv_m " + str(returned_nbr_mv_m) )
			
			nbr_mv_m_pixels_len = len( all_matches[nbr_mv_m_index][1] )
			
			# check if previous iterated nbr_mv_m_index  is neighbor movement with the current nbr_mv_m_index. if not, initialize the state.
			if returned_nbr_mv_m is not None:
				last_nbr_mv_m_indexes = _get_nbr_mv_matches( nbr_mv_m_index, {returned_nbr_mv_m} )
				if returned_nbr_mv_m not in last_nbr_mv_m_indexes:
					# returned_nbr_mv_m is not the movement neighbor with nbr_mv_m_index
					states["current_m_index"] = copy.deepcopy( states[cur_m_index] )
	
			# if the matched pixels length is going up 3 times or more, another real match is found. so get its highest pixels length.
			if states["current_m_index"]["going_back_up"] >= 3:
				if nbr_mv_m_pixels_len > states["current_m_index"]["cur_highest"]:
					states["current_m_index"]["cur_highest"] = nbr_mv_m_pixels_len
					states["current_m_index"]["cur_highest_m"] = nbr_mv_m_index
					#print("states['current_m_index']['cur_highest_m'] " + str(states["current_m_index"]["cur_highest_m"]) )

				else:
					states["current_m_index"]["going_back_down"] += 1
	
				if states["current_m_index"]["going_back_down"] >= 3:
					# now the direction is reversed.
					real_matches.add(states["current_m_index"]["cur_highest_m"])
					states["current_m_index"]["going_back_down"] = 0
					states["current_m_index"]["going_back_up"] = 0
					states["current_m_index"]["cur_lowest"] = nbr_mv_m_pixels_len
	
			else:
				if nbr_mv_m_pixels_len < states["current_m_index"]["cur_lowest"]:
					states["current_m_index"]["cur_lowest"] = nbr_mv_m_pixels_len
				elif nbr_mv_m_pixels_len > states["current_m_index"]["cur_lowest"]:
					# nbr_mv_m_pixels_len is bigger the last lowest. it is going back up.
					states["current_m_index"]["going_back_up"] += 1
					#print("states['current_m_index']['going_back_up'] added at nbr_mv_m_index " + str(nbr_mv_m_index) )

				if states["current_m_index"]["going_back_up"] >= 3:
					# here. the direction is reversed from going down to the going up. cur_lowest is the lowest for the current real match. now going up to find another real match
					states["current_m_index"]["cur_highest"] = nbr_mv_m_pixels_len
					states["current_m_index"]["cur_highest_m"] = nbr_mv_m_index

			# before calling the nested movement neighbor match index, update the states for nbr_mv_m_index.
			# before calling _get_real_matches, states[nbr_mv_m_index] and states[current_m_index] is exactly the same.
			# it will differ after calling it or when it returns from it.
			states[nbr_mv_m_index] = copy.deepcopy( states["current_m_index"] )
			#print("states[" + str(nbr_mv_m_index) + "] before calling _get_real_matches" )
			#print(states[nbr_mv_m_index])

			returned_nbr_mv_m = _get_real_matches( all_matches_in_sep_mv, nbr_mv_m_index, real_matches, done_m_indexes=done_m_indexes, states=states )
			
			if returned_nbr_mv_m is None:
				# no movement neighbor matches exist for the nbr_mv_m_index
				states["current_m_index"] = copy.deepcopy( states[nbr_mv_m_index] )
			else:
				# check if returned_nbr_mv_m is really the movement neighbor with nbr_mv_m_index
				last_nbr_mv_m_indexes = _get_nbr_mv_matches( nbr_mv_m_index, {returned_nbr_mv_m} )
				if returned_nbr_mv_m not in last_nbr_mv_m_indexes:
					# returned_nbr_mv_m is not the movement neighbor with nbr_mv_m_index
					# change the current_m_index states with the nbr_mv_m_index states.
					states["current_m_index"] = copy.deepcopy( states[nbr_mv_m_index] )

			#print("returned to " + str(nbr_mv_m_index) + " from the movements neighbors " + str(returned_nbr_mv_m) )
			#print("states['current_m_index'] " + str(states["current_m_index"] ) )


		return last_nbr_mv_m_index


	
	# { m_index: { match indexes that have neighbor movement }, ... }
	all_nbr_mvs = {}
	
	for m_index, each_match in enumerate(all_matches):
		# (matched_percent, matched_pixels, movement, out_im_pixels )
		
		cur_mv = each_match[2]
		
		x_minus1_mv = ( cur_mv[0] - 1, cur_mv[1] )
		x_plus1_mv = ( cur_mv[0] + 1, cur_mv[1] )
		y_minus1_mv = ( cur_mv[0], cur_mv[1] - 1 )
		y_plus1_mv = ( cur_mv[0], cur_mv[1] + 1 )
		
		cur_nbr_mvs = { x_minus1_mv, x_plus1_mv, y_minus1_mv, y_plus1_mv }
		
		all_nbr_mvs[m_index] = set()
		
		for ano_m_index, ano_match in enumerate(all_matches):
			if m_index == ano_m_index:
				continue
			
			if ano_match[2] in cur_nbr_mvs:
				all_nbr_mvs[m_index].add(ano_m_index)


	# all movements that are going up or going down 1 by 1.
	# [ { match indexes with movements that go up or down by 1},   ]
	all_matches_by_mv = _group_by_mv( all_nbr_mvs )

	all_real_matches = set()
	for all_matches_in_sep_mv in all_matches_by_mv:
		# all matches in separate movement.
		# { match indexes that are in the same sequential movements }
		
		# get the match with the largest matched pixels. this will be the real match in this movement sequence
		largest_m_index = _get_m_largest_mpixels( all_matches_in_sep_mv )

		# now iterate other matches from the largest_m_index by neighbor movements.
		real_matches = {largest_m_index}
		_get_real_matches( all_matches_in_sep_mv, largest_m_index, real_matches )
		
		all_real_matches |= real_matches

	
	best_m_index = _get_m_largest_mpixels(all_real_matches)
	
	return all_real_matches



# movements: list of movements. example. [ (3, 5), (4, 5), (5, 6), (8, 5), (8,4), (8,6), (1, 1) ]
# return:   [    [(3, 5), (4, 5), (5, 6)],     [(8, 5), (8, 4), (8, 6)],    [(1, 1)]    ]
def group_nbr_movements(movements):
    def are_neighbors(a, b):
        # Manhattan distance = 1 (up/down/left/right neighbors only)
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    def dfs(current, group, remaining):
        group.append(current)
        to_visit = []
        for other in remaining:
            if are_neighbors(current, other):
                to_visit.append(other)

        for other in to_visit:
            if other in remaining:  # ensure safety before removal
                remaining.remove(other)
                dfs(other, group, remaining)

    remaining = set(movements)  # use set for safe removal and O(1) lookup
    groups = []

    while remaining:
        current = remaining.pop()
        group = []
        dfs(current, group, remaining)
        groups.append(group)

    return groups



# all_matches:  [ (matched_percent, matched_pixels, (move_RL, move_UD), out_im_pixels ), ... ]
# all_matches are the return value from same_shapes_functions.match_by_mov_pixels_against_im3 
def get_real_matches_from_all_m2( all_matches ):

	all_m_mvs = { temp_match[2] for temp_match in all_matches }

	#  [ [(17, -12), (18, -12), ....],   [(4, 15), (4, 14), (5, 14), (5, 15)], ... ]
	all_mv_groups = group_nbr_movements(all_m_mvs)

	actual_matches = []
	for each_mv_group in all_mv_groups:
		cur_matches = [ temp_match for temp_match in all_matches if temp_match[2] in each_mv_group ]
		
		# get the highest match in current each_mv_group
		highest_in_mvgrp = max(cur_matches, key=lambda x: len(x[1] ) )
		actual_matches.append(highest_in_mvgrp)

	delete_indexes = set()
	for m_index, actual_match in enumerate(actual_matches):
		for ano_m_index, ano_actual_match in enumerate(actual_matches):
			if ano_m_index == m_index:
				continue
			
			overlap_pixels = actual_match[1].intersection(ano_actual_match[1] )
			overlap_percent = len(overlap_pixels) / len( actual_match[1] )
			if overlap_percent >= 0.3 and len( actual_match[1] ) < len( ano_actual_match[1] ):
				# delete actual_match
				delete_indexes.add(m_index)

	deleted = 0
	for delete_index in delete_indexes:
		actual_matches.pop( delete_index - deleted )
		deleted += 1

	return actual_matches



# result: [ [  { image1 pixels }, { matched image2 pixels }, matched movement ], ... ]
# matched image2 pixels are the direct matched pixels from image1 pixels but image1 pixels are not the direct matched pixels.
# so this function get direct matched image1 pixels from image2 direct matched pixels
def get_direct_matches( result, image_size ):

	for m_index, match in enumerate(result):
		im2to1_mv = ( match[2][0] * -1, match[2][1] * -1 )
		mv_pixels_at_im1 = pixel_functions.move_pixels( match[1], im2to1_mv[0], im2to1_mv[1], input_xy=True, im_size=image_size )
		dm_im1pixels = mv_pixels_at_im1.intersection( match[0] )
		
		result[m_index] = [ dm_im1pixels, match[1], match[2] ]

















































