
from libraries import pixel_shapes_functions, pixel_functions, image_functions, shapes_results_functions
from libraries.cv_globals import top_shapes_dir, top_images_dir
from libraries import cv_globals

from PIL import Image
import math
import os, sys, shutil
import copy
from collections import Counter
import time, numpy as np

m_tightness_threshold = 8

def check_shape_size( param_im1shapeid, param_im2shapeid, param_shapes, param_ano_shapes, size_threshold=1 ):
   # check if size is too different
   im1shapes_total = len( param_shapes[param_im1shapeid] )
   im2shapes_total = len( param_ano_shapes[param_im2shapeid] )
      
   im1_2pixels_diff = abs( im1shapes_total - im2shapes_total )
   if im1_2pixels_diff != 0:
      if im1shapes_total > im2shapes_total:
         if im1_2pixels_diff / im2shapes_total > size_threshold:
            return True
      else:
         if im1_2pixels_diff / im1shapes_total > size_threshold:
            return True

   return False      


# return True if size is too different, False if size is similar within threshold
def check_size_by_pixels( param_im1pixels, param_im2pixels, size_threshold=0.5, im_size=None  ):
   # check if size is too different
   im1shapes_total = len( param_im1pixels )
   im2shapes_total = len( param_im2pixels )
   
   im_pixels_size_factor = 1
   im_pixels_size_multiplier = 80
   # if im_size is provided, then this is a request to include the shape size in the calculation
   if im_size is not None:
      all_im_pixels_len = im_size[0] * im_size[1]
      im_pixels_average = ( im1shapes_total + im2shapes_total ) / 2
      # how many of im_pixels_average are there to fill the whole image?
      param_pix_need_to_fill = all_im_pixels_len / im_pixels_average
      # smaller the im_pixels_average is, the more is needed to fill the whole image.
      # bigger the im_pixels_average is, the size difference calculation should be more strict
      # the largest would result in 1. closer to 1, size calculation should be more strict
      im_pixels_size_factor = ( 1 / param_pix_need_to_fill ) * im_pixels_size_multiplier
      
      # the small enough pixels should be calculated only with this equation ( im1_2pixels_diff / im2shapes_total ).
      if im_pixels_size_factor < 1:
         im_pixels_size_factor = 1
      
   diff_value = None
   
   im1_2pixels_diff = abs( im1shapes_total - im2shapes_total )
   if im1_2pixels_diff != 0:
      if im1shapes_total > im2shapes_total:
         diff_value = ( im1_2pixels_diff / im2shapes_total ) * im_pixels_size_factor
      else:
         diff_value = ( im1_2pixels_diff / im1shapes_total ) * im_pixels_size_factor
      
      if diff_value > size_threshold:
         return True

   return False      


# size is different more than threshold -> returns True. False otherwise.
def size_diff_by_percent( pixels1, pixels2, threshold=0.5 ):

   if len(pixels1) > len(pixels2):
      bigger_pixels = pixels1
      smaller_pixels = pixels2
   else:
      bigger_pixels = pixels2
      smaller_pixels = pixels1
      
   if len(smaller_pixels) / len(bigger_pixels) < threshold:
      return True
   else:
      return False



# for same movement, True is returned.
def check_same_movement( movement1, movement2, threshold=5 ):
    
    min_threshold = 1
    penalty_value = 0
    for xy in range(2):

        x_or_y_diff = abs( movement1[xy] - movement2[xy] )
        max_x_or_y_value = max( [ abs(movement1[xy] ), abs( movement2[xy] ) ] )    
    
        x_or_y_threshold =  math.ceil( max_x_or_y_value / threshold )
        if x_or_y_threshold == 0:
            x_or_y_threshold = min_threshold

        penalty_value += x_or_y_diff - x_or_y_threshold
    
    if penalty_value >= 0:
        return False
    else:
        return True



# small shape can easily match same colored bigger shape. shape is a better match if more tightly the shape matches.
# boundary_input is true if parameter pixels is boundary pixels
# im_rgb_xy_data: { (x,y): (r,g,b), ... }
# return True if check is passed.
def check_match_tightness3( pixels, im_rgb_xy_data, im_size, min_colors, boundary_input=False ):
    
    # pixels need to be separeted into each connected ones for this properly work.
    shapes = pixel_functions.find_shapes2(pixels, im_rgb_xy_data, im_size, min_colors, input_xy=True)
    
    matched_pixels = set()
    for shapeid in shapes:
    
        if not boundary_input:
            boundary_pixels = pixel_shapes_functions.get_boundary_pixels(shapes[shapeid], im_size )
        else:
            boundary_pixels = shapes[shapeid]

        src_color = None
        for xy in shapes[shapeid]:
            src_color = im_rgb_xy_data[xy]
            break

        def _expand_boundary(param_boundary_p, image_size, expand_by=1 ):
            
            cur_layer = set(param_boundary_p)
            prev_layer_pixels = set(pixels)
            
            layer_constant = 0.18
            same_clr_total = 0
            
            for layer in range(1, expand_by + 1):
                layer_multiplier = layer_constant * layer
                
                new_layer_pixels = set()
                
                for pixel in cur_layer:
                    neighbors = pixel_functions.get_nbr_pixels(pixel, image_size, input_xy=True)
                    new_layer_pixels.update(neighbors)
                
                new_layer_pixels = new_layer_pixels.difference(prev_layer_pixels).difference(cur_layer)
                
                # now check how many pixels have the same color as the given parameter pixels.
                same_color_pixels = set()
                for new_ly_pixel in new_layer_pixels:
                    if min_colors:
                        if im_rgb_xy_data[new_ly_pixel] == src_color:
                            same_color_pixels.add(new_ly_pixel)
                    
                    else:
                        color_change = pixel_functions.compute_appearance_difference(im_rgb_xy_data[new_ly_pixel], src_color )
                        if not color_change:
                            same_color_pixels.add(new_ly_pixel)
                
                # lower layer number has more weights.
                same_clr_percent = len(same_color_pixels) / len(new_layer_pixels)
                same_clr_value = same_clr_percent * layer_multiplier
                same_clr_total += same_clr_value
                
                
                if layer == 1:
                    prev_layer_pixels.update(cur_layer)
                else:
                    prev_layer_pixels = cur_layer
                
                cur_layer = new_layer_pixels
                
                image_functions.cr_im_from_pixels( "26", "videos/street3/resized/min1/", prev_layer_pixels, pixels_rgb=(150,255,0) )
                image_functions.cr_im_from_pixels( "26", "videos/street3/resized/min1/", new_layer_pixels, pixels_rgb=(150,100,50), save_filepath=cv_globals.get_test_im_fpath() )
                image_functions.cr_im_from_pixels( "26", "videos/street3/resized/min1/", same_color_pixels, pixels_rgb=(50,200,50), input_im=cv_globals.get_test_im_fpath() )
            
            if same_clr_total >= 0.88:
                return False
            return True



        if _expand_boundary(boundary_pixels, im_size, expand_by=5 ):
            matched_pixels |= shapes[shapeid]

    match_percent = len(matched_pixels) / len( pixels )
    if match_percent < cv_globals.m_threshold:
        return False
    else:
        return True





# small shape can easily match same colored bigger shape. shape is a better match if more tightly the shape matches.
# boundary_input is true if parameter pixels is boundary pixels
# im_rgb_xy_data: { (x,y): (r,g,b), ... }
# return True if check is passed.
def check_match_tightness4( pixels, im_rgb_xy_data, im_size, min_colors, boundary_input=False, tightness_threshold=0.88 ):
    

    if not boundary_input:
        boundary_pixels = pixel_shapes_functions.get_boundary_pixels(pixels, im_size )
    else:
        boundary_pixels = shapes[shapeid]

    src_color = None
    for xy in pixels:
        src_color = im_rgb_xy_data[xy]
        break

    def _expand_boundary(param_boundary_p, image_size, expand_by=1 ):
        
        cur_layer = set(param_boundary_p)
        prev_layer_pixels = set(pixels)
        
        layer_constant = 0.18
        same_clr_total = 0
        
        for layer in range(1, expand_by + 1):
            layer_multiplier = layer_constant * layer
            
            new_layer_pixels = set()
            
            for pixel in cur_layer:
                neighbors = pixel_functions.get_nbr_pixels(pixel, image_size, input_xy=True)
                new_layer_pixels.update(neighbors)
            
            new_layer_pixels = new_layer_pixels.difference(prev_layer_pixels).difference(cur_layer)
            
            # now check how many pixels have the same color as the given parameter pixels.
            same_color_pixels = set()
            for new_ly_pixel in new_layer_pixels:
                if min_colors:
                    if im_rgb_xy_data[new_ly_pixel] == src_color:
                        same_color_pixels.add(new_ly_pixel)
                
                else:
                    color_change = pixel_functions.compute_appearance_difference(im_rgb_xy_data[new_ly_pixel], src_color )
                    if not color_change:
                        same_color_pixels.add(new_ly_pixel)
            
            # lower layer number has more weights.
            same_clr_percent = len(same_color_pixels) / len(new_layer_pixels)
            same_clr_value = same_clr_percent * layer_multiplier
            same_clr_total += same_clr_value
            
            
            if layer == 1:
                prev_layer_pixels.update(cur_layer)
            else:
                prev_layer_pixels = cur_layer
            
            cur_layer = new_layer_pixels
            
            #image_functions.cr_im_from_pixels( "26", "videos/street3/resized/min1/", prev_layer_pixels, pixels_rgb=(150,255,0) )
            #image_functions.cr_im_from_pixels( "26", "videos/street3/resized/min1/", new_layer_pixels, pixels_rgb=(150,100,50), save_filepath=cv_globals.get_test_im_fpath() )
            #image_functions.cr_im_from_pixels( "26", "videos/street3/resized/min1/", same_color_pixels, pixels_rgb=(50,200,50), input_im=cv_globals.get_test_im_fpath() )
        
        if same_clr_total >= tightness_threshold:
            return False
        return True



    return _expand_boundary(boundary_pixels, im_size, expand_by=5 )





# traverses all same color pixels outside the parameter "pixels" and return them.
def get_same_clr_conn_pixels(pixels, im_rgb_data, im_size, min_colors):
    boundary = pixel_shapes_functions.get_boundary_pixels(pixels, im_size=im_size)
    boundary = pixel_functions.convert_xy_pixels_to_pindexes(boundary, im_size)
    
    pixels_dict = {temp_pixel: True for temp_pixel in boundary}
    pixel_color = im_rgb_data[list(boundary)[0]]

    done_pixels = set()
    connected_pixels = set()
    
    for start_pixel in boundary:
        if start_pixel in done_pixels:
            continue
        
        stack = [start_pixel]
        
        while stack:
            current_pixel = stack.pop()
            
            if current_pixel in done_pixels:
                continue
            
            done_pixels.add(current_pixel)
            
            if current_pixel not in pixels_dict:
                # Only process if it's outside the original shape
                current_color = im_rgb_data[current_pixel]
                
                if min_colors:
                    if current_color != pixel_color:
                        continue
                else:
                    if pixel_functions.compute_appearance_difference(current_color, pixel_color):
                        continue
                
                connected_pixels.add(current_pixel)

            nbr_pixels = pixel_functions.get_nbr_pixels(current_pixel, im_size)
            for nbr in nbr_pixels:
                if nbr not in done_pixels:
                    stack.append(nbr)

    return connected_pixels



def match_pixels_against_imdata( pixels_to_match, im_data, im_shapeids_by_pindex, im_shapes, im_shapes_colors, im_size, min_colors, 
                                 return_all_matched=False, best_match=False, match_threshold=0.55, dbg_imfnums=None ):

   best_match_movement_values = None
   best_match_pixels = set()
   all_matched = []
   
   start_move = 0
   step_move = 1
   end_move = 20
   negative_start_move = -1
   # first one is for right and down. second tuple is for left and up.
   movements = [ ( start_move, end_move, step_move ), ( negative_start_move, -end_move, -step_move ) ]

   for movement in movements:

      for left_right in range( movement[0], movement[1], movement[2] ):
         
         for movement in movements:
            for down_up in range( movement[0], movement[1], movement[2] ):

               # { index: pixel color, ... }
               moved_pixels = {}
               for pindex in pixels_to_match:
                  xy = pixel_functions.convert_pindex_to_xy( pindex, im_size[0] )
                  moved_x = xy[0] + left_right
                  moved_y = xy[1] + down_up
                  
                  if moved_x < 0 or moved_x >= im_size[0] or moved_y < 0 or moved_y >= im_size[1]:
                     # out of image range
                     continue
               
                  moved_index = pindex + left_right + ( down_up * im_size[0] )
                  moved_pixels[ moved_index ] = pixels_to_match[pindex]

               
               matched_pixels_w_clrs = {}
               matched_pixels = set()
               for pindex in moved_pixels:

                  if min_colors is True:
                     if moved_pixels[pindex] == im_data[pindex]:
                        # color is the same
                        matched_pixels_w_clrs[pindex] = moved_pixels[pindex]
                        matched_pixels.add(pindex)
                  else:
                        appear_diff = pixel_functions.compute_appearance_difference( moved_pixels[pindex], im_data[pindex] )
                        if appear_diff is False:
                           # same appearance
                           matched_pixels_w_clrs[pindex] = moved_pixels[pindex]
                           matched_pixels.add(pindex)


               if len( matched_pixels_w_clrs ) / len( pixels_to_match ) >= match_threshold:
                  if best_match is True:                     
                     if len( matched_pixels_w_clrs ) > len( best_match_pixels ):
                     
                        # { pixels of the matched shapes }
                        matched_shapes_pixels = pixel_shapes_functions.get_matching_shapes_pixels( matched_pixels_w_clrs, im_shapeids_by_pindex, im_shapes, im_shapes_colors, min_colors, im_size )
                        if len(matched_shapes_pixels) >= 1:
                           print("dbg_imfnums " + str(dbg_imfnums) )
                           #image_functions.cr_im_from_pixels( dbg_imfnums[0], "videos/street3/resized/min1/", pixels_to_match, pixels_rgb=(100,0,255) ) 
                           #image_functions.cr_im_from_pixels( dbg_imfnums[1], "videos/street3/resized/min1/", matched_shapes_pixels, pixels_rgb=(255,255,0) ) 

                           m_tightness = check_match_tightness( matched_pixels, matched_shapes_pixels, im_size, len(moved_pixels) )
                           if m_tightness >= m_tightness_threshold:
                              continue
                        else:
                           continue
                        
                        # if almost the same matched_pixels length exist, then calculate the score taking in the fact that the closer the better match.
                        # for reference, refer to Documentations/libraries/same_shapes_functions/almost the same shape distance calculation.docx
                        if len(best_match_pixels) >= 1:
                           match = almost_same_shape_w_distance( len(matched_pixels), (left_right, down_up), len(best_match_pixels), best_match_movement_values )
                           if match is False:
                              continue

                        best_match_pixels = matched_pixels
                        best_match_movement_values = ( left_right, down_up )
                
                  elif return_all_matched is True:
                     all_matched.append( ( matched_pixels, ( left_right, down_up ) ) )
                  else:
                     return True
                  
   if best_match is True:
      return best_match_pixels, best_match_movement_values
   elif return_all_matched is True:
      return all_matched
   else:
      return False




# im1pixels and im2pixels are xy pixels.
# difference from match_shape_while_moving_it2 is the input pixels form.
# 
# # best_match means to not stop after finding the match at match_threshold but continue doing it and return the result of best match
def match_shape_while_moving_it3( im1pixels, im2pixels, im1_rgb_xy_data, im2_rgb_xy_data, im_size, min_colors, match_threshold=cv_globals.m_threshold, 
                                    best_match=False, return_all=False, debug=False, end_move=20, ignore_clrs=False ):
   
   im2pixels_lookup = {}
   for im2xy in im2pixels:
      im2pixels_lookup[im2xy] = True
   
   
   best_match_movement_values = None
   best_match_pixels = set()
   best_match_percent = 0
   all_matches = []
   
   sec_smallest_pixc = cv_globals.get_sec_smallest_pixc(im_size) # precompute
   
   start_move = 0
   step_move = 1
   negative_start_move = -1
   # first one is for right and down. second tuple is for left and up.
   movements = [ ( start_move, end_move, step_move ), ( negative_start_move, -end_move, -step_move ) ]
   
   for movement in movements:

      for left_right in range( movement[0], movement[1], movement[2] ):
         
         for movement in movements:
            for down_up in range( movement[0], movement[1], movement[2] ):

               # { moved xy: original color, ... }
               moved_im1pixels = {}
               out_im_pixels = set()
               
               for im1xy in im1pixels:

                  moved_x = im1xy[0] + left_right
                  moved_y = im1xy[1] + down_up

                  moved_xy = (moved_x, moved_y)
                  if moved_x < 0 or moved_x >= im_size[0] or moved_y < 0 or moved_y >= im_size[1]:
                     # out of image range
                     out_im_pixels.add( im1xy)
                     continue
               
                  if im2pixels_lookup.get(moved_xy): # this function only finds match within the provided parameter im2pixels
                     moved_im1pixels[moved_xy] = im1_rgb_xy_data[im1xy] # key is the moved pixel. value is original image1 color.

               # for out of image matching
               in_im_pixels_len = len(im1pixels) - len(out_im_pixels)
               remaining_percent = in_im_pixels_len / len(im1pixels)
               if remaining_percent < 0.15 or in_im_pixels_len < sec_smallest_pixc:
                  # at least 15% of the original pixels need to remain.
                  continue

               matched_pixels = set()
               if ignore_clrs is False:
                   for mv_pixel in moved_im1pixels:
                      if min_colors is True:
                         if moved_im1pixels[mv_pixel] == im2_rgb_xy_data[mv_pixel]:
                            # color is the same
                            matched_pixels.add( mv_pixel )
                      
                      else:
                         appear_diff = pixel_functions.compute_appearance_difference( moved_im1pixels[mv_pixel], im2_rgb_xy_data[mv_pixel] )
                         if appear_diff is False:
                            # same appearance
                            matched_pixels.add( mv_pixel )
                
               else:
                   matched_pixels = set( moved_im1pixels.keys() ).intersection(im2pixels)
                

               matched_percent = len(matched_pixels) / in_im_pixels_len
               
               #if ( 0, 0 ) == ( left_right, down_up ):
               #   image_functions.cr_im_from_pixels( '3', "videos/street4/resized/min1/", matched_pixels, pixels_rgb= (255, 255, 0 ) )
               
               if matched_percent >= match_threshold:

                  if return_all:
                     all_matches.append(  (matched_percent, matched_pixels, (left_right, down_up) ) )
               
                  elif best_match is True:                     
                     
                     if len(best_match_pixels) >= len(matched_pixels):
                        # failed
                        continue

                     best_match_movement_values = ( left_right, down_up )
                     best_match_pixels = matched_pixels
                     best_match_percent = matched_percent


                  else:
                     return True

   if return_all:
      return all_matches   
   
   elif best_match is True:
      return best_match_pixels, best_match_movement_values
   else:
      return False
        



# pixels -> xy pixels
# image data -> [ (255,0,0), ... ]. at pixel index is a color for that that pixel index.
# if threshold is provided, matching judgement is made inside the function and return matched pixels. matched pixels is empty if match was not made.
def match_pixels_at_specified_mvs( pixels, mvs, cur_image_data, opposite_im_data, im_size, min_colors, threshold=None, get_out_im=False ):

    most_mpixels = set()
    matched_mv = None

    for mv in mvs:

        matched_pixels = set()
        out_im_pixels = set()
        for pixel in pixels:
            
            mov_x = pixel[0] + mv[0]
            mov_y = pixel[1] + mv[1]
            
            if mov_x < 0 or mov_x >= im_size[0] or mov_y < 0 or mov_y >= im_size[1]:
                continue
            
            pixel_at_mv = (mov_x, mov_y)
            
            if pixel_at_mv is None:
                # out of image range
                out_im_pixels.add(pixel)
                continue
            
            if min_colors is True:
                if cur_image_data[pixel] == opposite_im_data[pixel_at_mv]:
                    matched_pixels.add(pixel_at_mv)

            else:
                appearance_diff = pixel_functions.compute_appearance_difference(cur_image_data[pixel], opposite_im_data[pixel_at_mv] )
                if appearance_diff is False:
                    # same color
                    matched_pixels.add(pixel_at_mv)
            
        if threshold is not None:
            # judge if match can be made

            in_im_pixels_len = len(pixels) - len(out_im_pixels)
            in_im_pixels_percent = in_im_pixels_len / len(pixels)
            if in_im_pixels_percent < 0.15 or in_im_pixels_len < cv_globals.get_sec_smallest_pixc(im_size ):
                # very few pixels remain
                continue
            
            matched_percent = len(matched_pixels) / in_im_pixels_len
            if matched_percent < threshold:
                continue
        
        if len(matched_pixels) > len(most_mpixels):
            most_mpixels = matched_pixels
            matched_mv = mv
        

    if not get_out_im:
        return most_mpixels, matched_mv
    else:
        return most_mpixels, matched_mv, out_im_pixels


# this is similair to match_shape_while_moving_it2. match_shape_while_moving_it2 takes 2 shapes and see if they match. match_shapes_against_image
# takes 2 shapes just like match_shape_while_moving_it2, but it also sees if image matches with the given shapes. This is created because 2 given
# shapes are both partially matching and match_shape_while_moving_it2 fails.
#
# pixels1 -> { pixel index: color, ... }. pixels1 will be used to match pixel with color.
# pixels2 -> { pixel indexes }. only pixel indexes are needed for pixels2. pixel color for the pixels2 will be matched from im_data.

def match_shapes_against_image( pixels1, pixels2, im_data, im_size, min_colors, match_threshold=0.55, dbg_imfnums=None ):

   best_match_score = 0
   best_match_movement = None
   best_match_pixels = set()
   all_matched = []
   
   start_move = 0
   step_move = 1
   end_move = 20
   negative_start_move = -1
   # first one is for right and down. second tuple is for left and up.
   movements = [ ( start_move, end_move, step_move ), ( negative_start_move, -end_move, -step_move ) ]

   for movement in movements:

      for left_right in range( movement[0], movement[1], movement[2] ):
         
         for movement in movements:
            for down_up in range( movement[0], movement[1], movement[2] ):

               # { index: pixel color, ... }
               moved_pixels = {}
               for pindex in pixels1:
                  xy = pixel_functions.convert_pindex_to_xy( pindex, im_size[0] )
                  moved_x = xy[0] + left_right
                  moved_y = xy[1] + down_up
                  
                  if moved_x < 0 or moved_x >= im_size[0] or moved_y < 0 or moved_y >= im_size[1]:
                     # out of image range
                     continue
               
                  moved_index = pindex + left_right + ( down_up * im_size[0] )
                  moved_pixels[ moved_index ] = pixels1[pindex]


               matched_pixels = set()
               for pindex in moved_pixels:

                  if min_colors is True:
                     if moved_pixels[pindex] == im_data[pindex]:
                        # color is the same
                        matched_pixels.add(pindex)
                  else:
                        appear_diff = pixel_functions.compute_appearance_difference( moved_pixels[pindex], im_data[pindex] )
                        if appear_diff is False:
                           # same appearance
                           matched_pixels.add(pindex)


               if len(moved_pixels) < 1 or len(matched_pixels) / len(moved_pixels) < match_threshold:
                   continue

               matched_im2pixels = matched_pixels.intersection( pixels2 )
               # matching image2 pixels have more weight than matching image pixels.
               cur_score = len(matched_pixels) + ( len(matched_im2pixels) * 1.3 )

               if cur_score > best_match_score:
                   best_match_score = cur_score
                   best_match_pixels = matched_pixels
                   best_match_movement = ( left_right, down_up )


   return best_match_pixels, best_match_movement


# out_im: out of image pixels.
# pixels_to_match are xy pixels.
def match_by_mov_pixels_against_im( pixels_to_match, src_im_data, target_im_data, im_size, min_colors, out_im=False, move_amount=16, m_threshold=None ):

    most_matched_pixels= set()
    most_matched_mv = None
    out_pixels = set()
    for pos_neg_RL in [ 1, -1 ]:
        if pos_neg_RL == 1:
            start_position_RL = 0
        else:
            start_position_RL = 1
   
        for move_RL in range( start_position_RL, move_amount ):
            move_RL *= pos_neg_RL
      
            for pos_neg_UD in [ 1, - 1 ]:
                if pos_neg_UD == 1:
                    start_position_UD = 0
                else:
                    start_position_UD = 1
         
                for move_UD in range( start_position_UD, move_amount ):
                    move_UD *= pos_neg_UD

                    matched_pixels = set()
                    cur_out_pixels = set()
                    for pixel in pixels_to_match:
                    
                        mov_x = pixel[0] + move_RL
                        mov_y = pixel[1] + move_UD
                        mv_pixel = (mov_x, mov_y)
                        
                        if mov_x < 0 or mov_x >= im_size[0] or mov_y < 0 or mov_y >= im_size[1]:
                            cur_out_pixels.add(pixel)
                            continue
                   
                        if min_colors is True:
                            if src_im_data[pixel] == target_im_data[mv_pixel]:
                                matched_pixels.add(mv_pixel)
                        
                        else:
                            appear_diff = pixel_functions.compute_appearance_difference( src_im_data[pixel], target_im_data[mv_pixel] )
                            if appear_diff is False:
                                # same appearance
                                matched_pixels.add(mv_pixel)

                    if not out_im:
                        matched_percent = len(matched_pixels) / len(pixels_to_match)
                        if m_threshold is not None and matched_percent < m_threshold:
                            continue
                        
                        if len(matched_pixels) > len(most_matched_pixels):
                            #match_tightness = check_match_tightness4( matched_pixels, target_im_data, im_size, min_colors, boundary_input=False )
                            #if not match_tightness:
                            #    continue
                            
                            most_matched_pixels = matched_pixels
                            most_matched_mv = (move_RL, move_UD)
                    
                    else:
                        
                        in_im_pixels_len = len(pixels_to_match) - len(cur_out_pixels)
                        in_im_pixels_percent = in_im_pixels_len / len(pixels_to_match)
                        if in_im_pixels_len < cv_globals.get_sec_smallest_pixc( im_size ) or in_im_pixels_percent < 0.15:
                            continue

                        matched_percent = len(matched_pixels) / in_im_pixels_len
                        if m_threshold is not None and matched_percent < m_threshold:
                            continue

                        if len(matched_pixels) > len(most_matched_pixels):
                            #match_tightness = check_match_tightness4( matched_pixels, target_im_data, im_size, min_colors, boundary_input=False )
                            #if not match_tightness:
                            #    continue

                            most_matched_pixels = matched_pixels
                            most_matched_mv = (move_RL, move_UD)
                            out_pixels = cur_out_pixels

    if not out_im:
        return most_matched_pixels, most_matched_mv
    else:
        return most_matched_pixels, most_matched_mv, out_pixels








# out_im: out of image pixels.
# difference from match_by_mov_pixels_against_im is that this returns all matches
def match_by_mov_pixels_against_im3( pixels_to_match, src_im_data, target_im_data, im_size, min_colors, out_im=False, threshold=0.55, check_tightness=False, move_amount=16 ):

    all_matches = []
    pixel_color = src_im_data[ list(pixels_to_match)[0] ]
   
    for pos_neg_RL in [ 1, -1 ]:
        if pos_neg_RL == 1:
            start_position_RL = 0
        else:
            start_position_RL = 1
   
        for move_RL in range( start_position_RL, move_amount ):
            move_RL *= pos_neg_RL
      
            for pos_neg_UD in [ 1, - 1 ]:
                if pos_neg_UD == 1:
                    start_position_UD = 0
                else:
                    start_position_UD = 1
         
                for move_UD in range( start_position_UD, move_amount ):
                    move_UD *= pos_neg_UD

                    matched_pixels = set()
                    out_im_pixels = set()
                    for pixel in pixels_to_match:
                        mov_x = pixel[0] + move_RL
                        mov_y = pixel[1] + move_UD
                        
                        mv_pixel = (mov_x, mov_y)
                        if mov_x < 0 or mov_x >= im_size[0] or mov_y < 0 or mov_y >= im_size[1]:
                            out_im_pixels.add(pixel)
                            continue
                   
                        if min_colors is True:
                            if src_im_data[pixel] == target_im_data[mv_pixel]:
                                matched_pixels.add(mv_pixel)
                        
                        else:
                            appear_diff = pixel_functions.compute_appearance_difference( src_im_data[pixel], target_im_data[mv_pixel] )
                            if appear_diff is False:
                                # same appearance
                                matched_pixels.add(mv_pixel)

                    if not out_im:
                        matched_percent = len(matched_pixels) / len(pixels_to_match)
                        if matched_percent >= threshold:
                            if check_tightness:
                                # check if matched_pixels are inside the big shape.
                                tighness_check = check_match_tightness3(matched_pixels, target_im_data, im_size, min_colors)
                                if tighness_check:
                                    all_matches.append( (matched_percent, matched_pixels, (move_RL, move_UD) ) )
                            
                            else:
                                all_matches.append( (matched_percent, matched_pixels, (move_RL, move_UD) ) )

                    else:
                        
                        in_im_pixels_len = len(pixels_to_match) - len(out_im_pixels)
                        in_im_pixels_percent = in_im_pixels_len / len(pixels_to_match)
                        if in_im_pixels_len < cv_globals.get_sec_smallest_pixc( im_size ) or in_im_pixels_percent < 0.15:
                            continue
                        
                        matched_percent = len(matched_pixels) / in_im_pixels_len
                        if matched_percent > threshold:
                            if check_tightness:
                                # check if matched_pixels are inside the big shape.
                                tighness_check = check_match_tightness3(matched_pixels, target_im_data, im_size, min_colors)
                                if tighness_check:
                                    all_matches.append( (matched_percent, matched_pixels, (move_RL, move_UD), out_im_pixels ) )
                            
                            else:
                                all_matches.append( (matched_percent, matched_pixels, (move_RL, move_UD), out_im_pixels ) )


    return all_matches



# match by excluding other movement matches.
# alrdy_m_nbrs: [ [ { image1 pixels }, { image2 pixels }, image1 to 2 movement ], ... ]
def match_by_excluding_oth_mvs(  expanded_pixels, image_size, im1_rgb_xy_data, im2_rgb_xy_data, min_colors, alrdy_m_nbrs, m_threshold=cv_globals.m_threshold, out_im=True, move_amount=16  ):

    sec_smallest_pixc = cv_globals.get_sec_smallest_pixc( image_size )

    most_matched_pixels= set()
    most_matched_mv = None
    out_pixels = set()
    for pos_neg_RL in [ 1, -1 ]:
        if pos_neg_RL == 1:
            start_position_RL = 0
        else:
            start_position_RL = 1
   
        for move_RL in range( start_position_RL, move_amount ):
            move_RL *= pos_neg_RL
      
            for pos_neg_UD in [ 1, - 1 ]:
                if pos_neg_UD == 1:
                    start_position_UD = 0
                else:
                    start_position_UD = 1
         
                for move_UD in range( start_position_UD, move_amount ):
                    move_UD *= pos_neg_UD

                    # exclude other movement neighbor matches from expanded_pixels
                    cur_mv_expanded_pixels = expanded_pixels.copy()
                    
                    for nbr_match in alrdy_m_nbrs:
                            same_movement = check_same_movement( nbr_match[2], (move_RL, move_UD), threshold=5 )
                            if same_movement is False:
                                cur_mv_expanded_pixels = cur_mv_expanded_pixels.difference(nbr_match[0] )

                    matched_pixels = set()
                    cur_out_pixels = set()
                    for pixel in cur_mv_expanded_pixels:
                    
                        mov_x = pixel[0] + move_RL
                        mov_y = pixel[1] + move_UD
                        mv_pixel = (mov_x, mov_y)
                        
                        if mov_x < 0 or mov_x >= image_size[0] or mov_y < 0 or mov_y >= image_size[1]:
                            cur_out_pixels.add(pixel)
                            continue
                   
                        if min_colors is True:
                            if im1_rgb_xy_data[pixel] == im2_rgb_xy_data[mv_pixel]:
                                matched_pixels.add(mv_pixel)
                        
                        else:
                            appear_diff = pixel_functions.compute_appearance_difference( im1_rgb_xy_data[pixel], im2_rgb_xy_data[mv_pixel] )
                            if appear_diff is False:
                                # same appearance
                                matched_pixels.add(mv_pixel)


                    if not out_im:
                        matched_percent = len(matched_pixels) / len(cur_mv_expanded_pixels)
                        if m_threshold is not None and matched_percent < m_threshold:
                            continue
                        
                        if len(matched_pixels) > len(most_matched_pixels):
                            #match_tightness = check_match_tightness4( matched_pixels, im2_rgb_xy_data, image_size, min_colors, boundary_input=False )
                            #if not match_tightness:
                            #    continue
                            
                            most_matched_pixels = matched_pixels
                            most_matched_mv = (move_RL, move_UD)
                    
                    else:
                        in_im_pixels_len = len(cur_mv_expanded_pixels) - len(cur_out_pixels)
                        in_im_pixels_percent = in_im_pixels_len / len(cur_mv_expanded_pixels)
                        if in_im_pixels_len < sec_smallest_pixc or in_im_pixels_percent < 0.15:
                            continue

                        matched_percent = len(matched_pixels) / in_im_pixels_len
                        if m_threshold is not None and matched_percent < m_threshold:
                            continue

                        if len(matched_pixels) > len(most_matched_pixels):
                            #match_tightness = check_match_tightness4( matched_pixels, im2_rgb_xy_data, image_size, min_colors, boundary_input=False )
                            #if not match_tightness:
                            #    continue

                            most_matched_pixels = matched_pixels
                            most_matched_mv = (move_RL, move_UD)
                            out_pixels = cur_out_pixels


    return most_matched_pixels, most_matched_mv



     
                    

# this gets surrounding pixels around given image1pixels and see if the whole pixels can match.
def match_with_expanded_pixels( image1pixels, image_size, input_xy, im1rgb_xy_data, im2rgb_xy_data, min_colors, threshold=cv_globals.m_threshold  ):

    im1expanded_pixels = pixel_functions.expand_boundary(image1pixels, image_size, expand_by=5, input_xy=input_xy)
    
    #image_functions.cr_im_from_pixels( image_fname, directory, im1expanded_pixels , pixels_rgb=(255,0,0) )
    
    expanded_m_pixels, m_mv, out_pixels = match_by_mov_pixels_against_im( im1expanded_pixels, im1rgb_xy_data, im2rgb_xy_data, image_size, min_colors, out_im=True )
    in_im_pixels_len = len(im1expanded_pixels) - len(out_pixels)

    if in_im_pixels_len < cv_globals.get_sec_smallest_pixc(image_size) or in_im_pixels_len / len(im1expanded_pixels) < 0.15: # too many pixels went out of image
        return None, None

    m_percent = len(expanded_m_pixels) / in_im_pixels_len
    if m_percent < threshold:
        return None, None

    target_in_im_pixels = image1pixels.difference(out_pixels)

    pixels_at_im2 = pixel_functions.move_pixels( image1pixels, m_mv[0], m_mv[1], input_xy=input_xy, im_size=image_size )
    target_m_pixels = pixels_at_im2.intersection(expanded_m_pixels)
    target_m_percent = len(target_m_pixels) / len(target_in_im_pixels)

    if target_m_percent < threshold:
        return None, None
    
    return target_m_pixels, m_mv
    
    #image_functions.cr_im_from_pixels( image2_fname, directory, test_m_pixels , pixels_rgb=(0,0,255) )


# split and match 2shapes
# shape2 is the combined pixels that includes shape1_im1pixels
def split_n_match_2shapes(  shape1_im1pixels, shape2_im1pixels, shape2_im2pixels, im1_rgb_xy_data, im2_rgb_xy_data, im_size, min_colors ):

    s1_im1pixels = shape1_im1pixels.intersection(shape2_im1pixels)
    
    s1_im1pixels, s2_im1pixels = pixel_shapes_functions.split_2shapes_in_combined_S( s1_im1pixels, shape2_im1pixels, im_size )
    
    
    # [   ( matched percent , {matched pixels}, movement ), ... ]
    s1_all_matches = match_shape_while_moving_it3( s1_im1pixels, shape2_im2pixels, im1_rgb_xy_data, im2_rgb_xy_data, im_size, min_colors, return_all=True )
    s2_all_matches = match_shape_while_moving_it3( s2_im1pixels, shape2_im2pixels, im1_rgb_xy_data, im2_rgb_xy_data, im_size, min_colors, return_all=True )
    
    # both shapes all matches are taken. both shapes should have the most matched pixels with the LEAST overlapping pixels between the both shapes matches.
    
    best_score = 0
    best_match = None  # ( shape1 match index, shape2 match index )
    for s1m_index, s1_match in enumerate(s1_all_matches):
        # ( matched percent , {matched pixels}, movement )
        
        s1_matched_pixels = s1_match[1]
        
        for s2m_index, s2_match in enumerate(s2_all_matches):
            
            s2_matched_pixels = s2_match[1]
            
            ovlp_pixels = s1_matched_pixels.intersection(s2_matched_pixels)
            s1_ovlp_percent = len(ovlp_pixels) / len(s1_matched_pixels)
            s2_ovlp_percent = len(ovlp_pixels) / len(s2_matched_pixels)
            
            if s1_ovlp_percent >= 0.25 or s2_ovlp_percent >= 0.25:
                continue
            
            score = len(s1_matched_pixels) + len(s2_matched_pixels)
            
            if score > best_score:
                best_match = ( s1m_index, s2m_index )
                best_score = score
    
    
    if best_match is None:
        return None, None
    
    s1_best_match = s1_all_matches[best_match[0] ]
    s1_best_m_pixels = s1_best_match[1]
    s1_best_m_mv = s1_best_match[2]

    s2_best_match = s2_all_matches[best_match[1] ]
    s2_best_m_pixels = s2_best_match[1]
    s2_best_m_mv = s2_best_match[2]
    

    # both shapes best match pixels are obtained. but this is not the end. there may still be some leftover pixels that are not in best match pixels.
    leftover_pixels = shape2_im2pixels.difference( s1_best_m_pixels )
    leftover_pixels = leftover_pixels.difference( s2_best_m_pixels )
    
    leftover_shapes = pixel_functions.find_shapes(leftover_pixels, im_size, input_xy=True )
    
    for leftover_shapeid in leftover_shapes:
        # now belong the shape to the closest shape, shape1 or shape2.
        
        s1_distance = pixel_shapes_functions.get_closest_pixels( s1_best_m_pixels, leftover_shapes[leftover_shapeid], im_size=im_size, return_distance=True )
        s2_distance = pixel_shapes_functions.get_closest_pixels( s2_best_m_pixels, leftover_shapes[leftover_shapeid], im_size=im_size, return_distance=True )
        
        if s1_distance < s2_distance:
            # shape1 is closer, put leftover shape to the shape1
            s1_best_m_pixels |= leftover_shapes[leftover_shapeid]
        
        elif s1_distance > s2_distance:
            s2_best_m_pixels |= leftover_shapes[leftover_shapeid]

        else:
            # both are same distance
            s1_best_m_pixels |= leftover_shapes[leftover_shapeid]
            s2_best_m_pixels |= leftover_shapes[leftover_shapeid]
    
    return ( shape1_im1pixels, s1_best_m_pixels, s1_best_m_mv ), ( s2_im1pixels, s2_best_m_pixels, s2_best_m_mv )
                

            

# split and match 2shapes
# shape2 is the combined pixels that includes multi_s1_shapes       
def split_n_match_multi_shapes(  multi_s1_shapes, shape2_im1pixels, shape2_im2pixels, im1_rgb_xy_data, im2_rgb_xy_data, im_size, min_colors ):
    
    # s2_im1pixels are now pixels left after subtracting multi_s1_shapes.
    ovlp_multi_s1_shapes, s2_im1_biggest_s = pixel_shapes_functions.split_multi_shapes_in_combined_S( multi_s1_shapes, shape2_im1pixels, im_size )
    
    all_im1_shapes = [s2_im1_biggest_s]
    all_im1_shapes.extend(ovlp_multi_s1_shapes)

    all_shapes_matches = []
    for im1_shape in all_im1_shapes:
        # [   ( matched percent , {matched pixels}, movement ), ... ]
        shape_matches = match_shape_while_moving_it3( im1_shape, shape2_im2pixels, im1_rgb_xy_data, im2_rgb_xy_data, im_size, min_colors, return_all=True )
        all_shapes_matches.append(shape_matches)
    
    
    no_ovlp_shapes_matches = {}   # { shape index: best match, ... }
    for shape_m_index, shape_matches in enumerate(all_shapes_matches):
        best_match = None
        best_match_len = 0
        
        for s_m_index, shape_match in enumerate(shape_matches):
            check_overlap = True
            
            for ano_shape_m_index, ano_shape_matches in enumerate(all_shapes_matches):
                if shape_m_index == ano_shape_m_index:
                    continue
                
                for ano_s_m_index, ano_s_match in enumerate(ano_shape_matches):
                
                    overlap_pixels = shape_match[1].intersection( ano_s_match[1] )
                    shape_ovlp_percent = len(overlap_pixels) / len(shape_match[1] )
                    ano_s_ovlp_percent = len(overlap_pixels) / len(ano_s_match[1] )
                    
                    if shape_ovlp_percent >= 0.25 or ano_s_ovlp_percent >= 0.25:
                        # overlap happened. current shape_match failed
                        check_overlap = False
                        break
                
                if check_overlap is False:
                    break
            
            if check_overlap:
                # no other overlaps with any other shapes matches.
                
                if len(shape_match[1] ) > best_match_len:
                    best_match = shape_match
                    best_match_len = len( shape_match[1] )
        
        no_ovlp_shapes_matches[shape_m_index] = best_match
    
    
    s2_best_match = no_ovlp_shapes_matches.pop(0)
    s1_best_matches = list( no_ovlp_shapes_matches.values() )
    
    return s1_best_matches, s2_best_match



# pixch: given pixels' pixch. in this function, pixch indicate the shape's movement. if the pixch is the result of deformation, then this function will not work.
# at_mvs: at specifict movements. the value for this is the specific movements. this is to only try these movements. not all movements upto move_amount
def match_pixch_shape2( pixels, pixch, from_im_rgb_data, to_match_im_rgb_data, im_size, min_colors, move_amount=16, m_threshold=cv_globals.m_threshold, return_all_m=False, at_mvs=None ):

    nonpixch = pixels.difference(pixch)
    
    # for pixel change more than 50%. nonpixch will not be used
    pixch_percent = len(pixch) / len(pixels)

    sec_smallest_pixc = cv_globals.get_sec_smallest_pixc(im_size)

    most_matched_pixels= set()
    best_m_percent = 0
    all_matches = []    # all matches with matched pixels above threshold.
    
    most_matched_mv = None
    out_pixels = set()
    for pos_neg_RL in [ 1, -1 ]:
        if pos_neg_RL == 1:
            start_position_RL = 0
        else:
            start_position_RL = 1
   
        for move_RL in range( start_position_RL, move_amount ):
            move_RL *= pos_neg_RL
      
            for pos_neg_UD in [ 1, - 1 ]:
                if pos_neg_UD == 1:
                    start_position_UD = 0
                else:
                    start_position_UD = 1
         
                for move_UD in range( start_position_UD, move_amount ):
                    move_UD *= pos_neg_UD
                    
                    if at_mvs is not None and ( move_RL, move_UD) not in at_mvs:
                        continue
                    
                    moved_pixels = set()  # this is just for debugging purpose
                    matched_pixels = set()   # this will be matched pixels with nonpixch_im2pixels
                    out_im_pixels = set()
                    for pixel in pixels:
                    
                        mov_x = pixel[0] + move_RL
                        mov_y = pixel[1] + move_UD
                        mv_pixel = (mov_x, mov_y)
                        
                        if mov_x < 0 or mov_x >= im_size[0] or mov_y < 0 or mov_y >= im_size[1]:
                            out_im_pixels.add(pixel)
                            continue
                        
                        moved_pixels.add(mv_pixel)
                        
                        if min_colors is True:
                            if from_im_rgb_data[pixel] == to_match_im_rgb_data[mv_pixel]:
                                matched_pixels.add(mv_pixel)
                        
                        else:
                            appear_diff = pixel_functions.compute_appearance_difference( from_im_rgb_data[pixel], to_match_im_rgb_data[mv_pixel] )
                            if appear_diff is False:
                                # same appearance
                                matched_pixels.add(mv_pixel)


                    #image_functions.cr_im_from_pixels( '2', "videos/street4/resized/min1/",    matched_pixels,  pixels_rgb=(0,0,  255) )

                    still_remain_pixch = pixch.intersection(moved_pixels)

                    if pixch_percent < 0.5:
                        nonpixch_ovlp = matched_pixels.intersection(nonpixch)
                        nonpixch_percent = len(nonpixch_ovlp) / len(nonpixch)  # the larger the better
                        if nonpixch_percent < m_threshold:
                            continue
                    else:
                        nonpixch_percent = 0  # for pixel change more than 50%. nonpixch will not be used

                    if len(pixch) == 0:
                        remain_pixch_percent = 0
                    else:   
                        remain_pixch_percent = len(still_remain_pixch) / len(pixch)  # the smaller the better.
                        if remain_pixch_percent >= 1 - m_threshold:
                            continue
                    
                    in_im_pixels_len = len(pixels) - len(out_im_pixels)
                    in_im_pixels_percent = in_im_pixels_len / len(pixels)
                    if in_im_pixels_len < sec_smallest_pixc or in_im_pixels_percent < 0.15:
                        continue
                    
                    matched_percent = len(matched_pixels) / in_im_pixels_len
                    if matched_percent < m_threshold: 
                        continue

                    real_m_percent = len(matched_pixels) / len(pixels)    # of course it is more favorable to have more matched pixels.
                    total_percent = ( real_m_percent * 1.2 ) + nonpixch_percent - remain_pixch_percent
                    
                    if return_all_m is True:
                        all_matches.append( ( round(total_percent, 2), matched_pixels, (move_RL, move_UD) ) )

                    if total_percent > best_m_percent:

                        most_matched_pixels = matched_pixels
                        most_matched_mv = (move_RL, move_UD)
                        best_m_percent = total_percent

    if return_all_m is False:
        return most_matched_pixels, most_matched_mv
    elif return_all_m is True:
        return all_matches



# pixch: given pixels' pixch. in this function, pixch indicate the shape's movement. if the pixch is the result of deformation, then this function will not work.
# at_mvs: at specifict movements. the value for this is the specific movements. this is to only try these movements. not all movements upto move_amount
def match_G50pixch_shapes( shape_pixels, shape_pixch, btwn_im_pixch, im_size, min_colors, from_im_rgb_data, to_im_rgb_data, move_amount=22, m_threshold=cv_globals.m_threshold, at_mvs=None ):

    sec_smallest_pixc = cv_globals.get_sec_smallest_pixc(im_size)

    most_matched_pixels= set()
    best_m_percent = 0
    
    most_matched_mv = None
    out_pixels = set()
    for pos_neg_RL in [ 1, -1 ]:
        if pos_neg_RL == 1:
            start_position_RL = 0
        else:
            start_position_RL = 1
   
        for move_RL in range( start_position_RL, move_amount ):
            move_RL *= pos_neg_RL
      
            for pos_neg_UD in [ 1, - 1 ]:
                if pos_neg_UD == 1:
                    start_position_UD = 0
                else:
                    start_position_UD = 1
         
                for move_UD in range( start_position_UD, move_amount ):
                    move_UD *= pos_neg_UD
                    
                    if at_mvs is not None and ( move_RL, move_UD) not in at_mvs:
                        continue
                    
                    #if move_RL < -2 or move_RL > 2 or move_UD < -2 or move_UD > 2: # restricting movement for debugging. remove after use.
                    #    continue
                    
                    matched_pixels = set()
                    moved_pixels = set()
                    out_pixels = set()
                    for pixel in shape_pixels:
                    
                        moved_x = pixel[0] + move_RL
                        moved_y = pixel[1] + move_UD
                        
                        if moved_x < 0 or moved_x >= im_size[0] or moved_y < 0 or moved_y >= im_size[1]:
                            out_pixels.add(pixel)
                            continue
                        
                        moved_pixels.add( (moved_x, moved_y) )
                        
                        if min_colors is True:
                            if from_im_rgb_data[pixel] == to_im_rgb_data[ (moved_x, moved_y) ]:
                                matched_pixels.add( (moved_x, moved_y) )
                        
                        else:
                            clr_diff = pixel_functions.compute_appearance_difference(from_im_rgb_data[pixel], to_im_rgb_data[ (moved_x, moved_y) ] )
                            if clr_diff is False:
                                matched_pixels.add( (moved_x, moved_y) )
                    
                    moved_out_pixels = moved_pixels.difference(shape_pixels)
                    matched_pixch = matched_pixels.intersection(btwn_im_pixch)
                    matched_pixch_percent = len(matched_pixch) / len(moved_out_pixels) if len(moved_out_pixels) > 0 else 0

                    #image_functions.cr_im_from_pixels( '2', "videos/street4/resized/min1/",    matched_pixels,  pixels_rgb=(0,0,  255) )

                    still_remain_pixch = shape_pixch.intersection(moved_pixels)

                    in_im_pixels_len = len(shape_pixels) - len(out_pixels)
                    in_im_pixels_percent = in_im_pixels_len / len(shape_pixels)
                    if in_im_pixels_len < sec_smallest_pixc or in_im_pixels_percent < 0.15:
                        continue    # not enough pixels remain
                    
                    matched_percent = len(matched_pixels) / in_im_pixels_len
                    if matched_percent < m_threshold: 
                        #print("matched_percent " + str(matched_percent) )
                        continue    # pixel matching failed

                    if len(shape_pixch) == 0:
                        remain_pixch_percent = 0
                    else:   
                        remain_pixch_percent = len(still_remain_pixch) / len(shape_pixch)  # the smaller the better.
                    
                    # below condition indicates that the shape should move to match
                    # remain_pixch_percent: small.  matched pixels: high.  matched_pixch_percent is high. 
                    # below indicates that the shape may has deformed.
                    # remain_pixch_percent is above medium. matched pixels: medium to high. matched_pixch_percent is small but pixel changes are in other areas near the shape.
                    
                    # this function only deals with the shape movement matches.
                    if remain_pixch_percent >= 1 - m_threshold or matched_pixch_percent < 0.55 :
                        #print("remain_pixch_percent " + str(remain_pixch_percent) + " matched_pixch_percent " + str(matched_pixch_percent) )
                        continue    # failed shape movement matching.


                    real_m_percent = len(matched_pixels) / len(shape_pixels)    # of course it is more favorable to have more matched shape_pixels.
                    total_percent = ( real_m_percent * 1.2 ) + matched_pixch_percent - remain_pixch_percent
                    if total_percent > best_m_percent:

                        most_matched_pixels = matched_pixels
                        most_matched_mv = (move_RL, move_UD)
                        best_m_percent = total_percent

    return most_matched_pixels, most_matched_mv




# pixch: given pixels' pixch. in this function, pixch indicate the shape's movement. if the pixch is the result of deformation, then this function will not work.
# at_mvs: at specifict movements. the value for this is the specific movements. this is to only try these movements. not all movements upto move_amount
def match_L50pixch_shapes( shape_pixels, shape_pixch, all_to_im_pixels,
                                im_size, min_colors, move_amount=16, m_threshold=cv_globals.m_threshold, at_mvs=None ):

    nonpixch = shape_pixels.difference(shape_pixch)

    sec_smallest_pixc = cv_globals.get_sec_smallest_pixc(im_size)

    most_matched_pixels= set()
    best_m_percent = 0
    
    most_matched_mv = None
    out_pixels = set()
    for pos_neg_RL in [ 1, -1 ]:
        if pos_neg_RL == 1:
            start_position_RL = 0
        else:
            start_position_RL = 1
   
        for move_RL in range( start_position_RL, move_amount ):
            move_RL *= pos_neg_RL
      
            for pos_neg_UD in [ 1, - 1 ]:
                if pos_neg_UD == 1:
                    start_position_UD = 0
                else:
                    start_position_UD = 1
         
                for move_UD in range( start_position_UD, move_amount ):
                    move_UD *= pos_neg_UD
                    
                    if at_mvs is not None and ( move_RL, move_UD) not in at_mvs:
                        continue
                    
                    #if move_RL < -2 or move_RL > 2 or move_UD < -2 or move_UD > 2: # restricting movement for debugging. remove after use.
                    #    continue
                    
                    moved_pixels, out_im_pixels = pixel_functions.move_pixels( shape_pixels, move_RL, move_UD, input_xy=True, im_size=im_size, get_out_of_im=True )
                    matched_pixels = moved_pixels.intersection(all_to_im_pixels)

                    nonpixch_ovlp = matched_pixels.intersection(nonpixch)
                    nonpixch_percent = len(nonpixch_ovlp) / len(nonpixch) if len(nonpixch) > 0 else 0   # the larger the better
                    if nonpixch_percent < m_threshold:
                        #print("nonpixch_percent " + str(nonpixch_percent) )
                        continue # non-pixch pixels should match


                    in_im_pixels_len = len(shape_pixels) - len(out_im_pixels)
                    in_im_pixels_percent = in_im_pixels_len / len(shape_pixels)
                    if in_im_pixels_len < sec_smallest_pixc or in_im_pixels_percent < 0.15:
                        continue    # not enough pixels remain
                    
                    matched_percent = len(matched_pixels) / in_im_pixels_len
                    if matched_percent < m_threshold: 
                        #print("matched_percent " + str(matched_percent) )
                        continue    # pixel matching failed

                    real_m_percent = len(matched_pixels) / len(shape_pixels)    # of course it is more favorable to have more matched shape_pixels.
                    total_percent = ( real_m_percent * 1.2 ) + nonpixch_percent

                    if total_percent > best_m_percent:

                        most_matched_pixels = matched_pixels
                        most_matched_mv = (move_RL, move_UD)
                        best_m_percent = total_percent


    return most_matched_pixels, most_matched_mv




# shapes_pixels: [ { shape pixels}, ... ] 
# if matched pixels have btwn_im_pixch, those may be the deformed pixels. if matched pixels do not contain the btwn_im_pixch, they may be the stayed intact pixels.
def match_deformed_shapes( shapes_pixels, btwn_im_pixch, start_mv_pixel, end_mv_pixel, im1_rgb_xy_data, im2_rgb_xy_data, 
                            im1_xy_shapeids_by_p, im2_xy_shapeids_by_p, image_size, min_colors, m_threshold=0.4  ):
    
    if os.path.exists( top_shapes_dir + "videos/street4/resized/min1/temp/test.png" ):
        os.remove( top_shapes_dir + "videos/street4/resized/min1/temp/test.png" )
    
    # shapes with expanded pixels: shape matching needs to be confirmed by neighbor shapes.
    # [ (shape pixels, boundary expanded pixels), ... ]
    shapes_w_exp = []
    
    for shape_pixels in shapes_pixels:
        # each shape matching needs to be confirmed by surrounding shapes.
        boundary_pixels = pixel_shapes_functions.get_boundary_pixels(shape_pixels, im_size=image_size)
        expanded_pixels = pixel_functions.expand_boundary(boundary_pixels, image_size, expand_by=2, input_xy=True)
        expanded_pixels = expanded_pixels.difference(shape_pixels)
        
        shapes_w_exp.append( (shape_pixels, expanded_pixels ) )
    
    second_sml_pixc = cv_globals.get_sec_smallest_pixc(image_size)
    
    def_match_threshold = 0.35 # match threshold for deformed shapes
    
    matched_shapes_result = [[ 0, set(), None] for _ in range( len(shapes_pixels) )] # create list with the same size as shapes_pixels. 
                                                                                 # each element is a shape with expanded match percent, matched pixels and matched movement


    def _match_pixel(pixel, im_out_pixels, matched_pixels  ):

        move_x = pixel[0] + move_RL
        move_y = pixel[1] + move_UD
        
        if move_x < 0 or move_x >= image_size[0] or move_y < 0 or move_y >= image_size[1]:
            im_out_pixels.add(pixel)
            return
        
        if min_colors is True:
            if im1_rgb_xy_data[pixel] == im2_rgb_xy_data[ (move_x, move_y) ]:
                matched_pixels.add( (move_x, move_y) )
        
        else:
            color_diff = pixel_functions.compute_appearance_difference(im1_rgb_xy_data[pixel], im2_rgb_xy_data[ (move_x, move_y) ] )
            if color_diff is False:
                # color is the same
                matched_pixels.add( (move_x, move_y) )


    for move_RL in range( start_mv_pixel[0], end_mv_pixel[0] + 1):
        for move_UD in range( start_mv_pixel[1], end_mv_pixel[1] + 1):
            
            for shape_index, shape_w_exp in enumerate(shapes_w_exp ):
                shape_matched_pixels = set()
                shape_im_out_pixels = set()
                
                exp_matched_pixels = set()
                exp_im_out_pixels = set()
                
                for shape_pixel in shape_w_exp[0]:
                    _match_pixel(shape_pixel, shape_im_out_pixels, shape_matched_pixels )
                
                if len(shape_matched_pixels) == 0:
                    continue
                
                for expanded_pixel in shape_w_exp[1]:
                    _match_pixel(expanded_pixel, exp_im_out_pixels, exp_matched_pixels )
                
                in_im_exp_len = len( shape_w_exp[1] ) - len(exp_im_out_pixels)
                in_im_exp_percent = in_im_exp_len / len( shape_w_exp[1] )
                if in_im_exp_len < second_sml_pixc or in_im_exp_percent < 0.15:
                    continue

                in_im_exp_mpercent = len(exp_matched_pixels) / in_im_exp_len

                '''
                if shape_index != 0 or move_UD != 7:
                    continue
                print("move_RL " + str(move_RL) + " move_UD " + str(move_UD) + " len(shape_matched_pixels) " + str( len(shape_matched_pixels) ) + " in_im_exp_mpercent " + str(in_im_exp_mpercent) )
                exp_debug_nbrs = set()
                for pixel in shape_w_exp[1]:
                    exp_debug_nbrs |= pixel_functions.get_nbr_pixels( pixel, image_size, input_xy=True)
                exp_debug_nbrs = exp_debug_nbrs.difference( shape_w_exp[1] ).difference( shape_w_exp[0] )
                
                if not os.path.exists( top_shapes_dir + "videos/street4/resized/min1/temp/test.png" ):
                    image_functions.cr_im_from_pixels( '3', "videos/street4/resized/min1/", shape_w_exp[0] , pixels_rgb=(255, 0,0), save_filepath=top_shapes_dir + "videos/street4/resized/min1/temp/test.png" )
                    image_functions.cr_im_from_pixels( '3', "videos/street4/resized/min1/", exp_debug_nbrs, pixels_rgb=(255, 255,0), input_im=top_shapes_dir + "videos/street4/resized/min1/temp/test.png" )
                
                im2_expanded_pixels = pixel_functions.move_pixels( shape_w_exp[1], move_RL, move_UD, input_xy=True, im_size=image_size )
                
                exp_debug_nbrs = set()
                for pixel in im2_expanded_pixels:
                    exp_debug_nbrs |= pixel_functions.get_nbr_pixels( pixel, image_size, input_xy=True)
                exp_debug_nbrs = exp_debug_nbrs.difference(im2_expanded_pixels )
                
                image_functions.cr_im_from_pixels( '4', "videos/street4/resized/min1/", exp_debug_nbrs, pixels_rgb=(0, 255, 255) )
                '''
                
                if in_im_exp_mpercent < m_threshold:
                    continue
                
                if in_im_exp_mpercent > matched_shapes_result[shape_index][0]:
                    matched_shapes_result[shape_index] = [ in_im_exp_mpercent, shape_matched_pixels, (move_RL, move_UD) ]

    print("matched_shapes_result")
    print(matched_shapes_result)

    return matched_shapes_result
    
















    
                    


