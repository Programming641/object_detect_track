
from collections import Counter
from PIL import Image
import os, sys
from statistics import mean
import math
import pickle

from libraries import image_functions, pixel_functions, same_shapes_functions
from libraries.cv_globals import proj_dir, top_images_dir, top_shapes_dir

# im_size is just there for backward compatibility.
def get_boundary_pixels(pixels, im_size=None):
    pixels = set(pixels)  # ensure it's a set for fast lookup
    boundary = set()

    for (x, y) in pixels:
        # Check 4-connected neighbors
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        for nx, ny in neighbors:
            if (nx, ny) not in pixels:
                boundary.add((x, y))
                break  # no need to check more neighbors, already boundary
    return boundary



# for explanation, see documentation, libraries/same_shapes_functions/check_match_from_possible_matches.docx. section title: how can I get pixels on moved side?
# pixels -> list or set of pixel indexes or xy pixels.
# x_or_y -> "x" or "y". if x is provided, on every y, get all consecutive x. if y is provided, on every x, get all consecutive y.
#
# return: { x or y: [ set of opposite x or y values in one consecutive, ... ], ... }. with sample data { 15: [ {1,3,2}, {8,9,7}, ..., }
def get_consec_pixels_on_x_or_y( pixels, x_or_y, im_size ):

    if x_or_y == "y":
        xy_to_get = 1
        on_every_xy = 0
    else:
        xy_to_get = 0
        on_every_xy = 1
    
    pixels_xy = set()
    if type( list(pixels)[0] ) is int:
        for pindex in pixels:
            xy = pixel_functions.convert_pindex_to_xy( pindex, im_size[0] )
            pixels_xy.add(xy)
    else:
        pixels_xy = pixels
    
    on_every_xy_values = { temp_xy[on_every_xy] for temp_xy in pixels_xy }
    

    def get_smallest_consecutive( cur_all_values ):
        smallest_consecutive = set()
        
        prev_value = None
        for cur_value in cur_all_values:
            
            if prev_value is None:
                prev_value = cur_value
                smallest_consecutive.add(prev_value)
            elif cur_value != prev_value + 1:
                # current smallest_consecutive has ended.
                return smallest_consecutive
    
            else:
                smallest_consecutive.add(cur_value)
                prev_value = cur_value
                
    
        # all values of cur_all_values are contained in smallest_consecutive
        return smallest_consecutive
    
    
    all_consecutive_x_or_y = {}
    for current_on_xy in on_every_xy_values:
        # [ set of pixels in one consecutive for current_on_xy, ... ]
        cur_xy_result = []
        
        # get all x or y values for the current_on_xy
        cur_all_xy = [ temp_xy[xy_to_get] for temp_xy in pixels_xy if temp_xy[on_every_xy] == current_on_xy ]
        
        # now I need to get every consecutive y
        while len( cur_all_xy ) >= 1:
            cur_all_xy.sort()
            cur_smallest_consec = get_smallest_consecutive( cur_all_xy )

            cur_xy_result.append(cur_smallest_consec)
            
            cur_all_xy = list( set(cur_all_xy).difference(cur_smallest_consec) )

        all_consecutive_x_or_y[current_on_xy] = cur_xy_result

    return all_consecutive_x_or_y



def get_shape_color( shape_pixels, im_rgb_data, min_colors ):

    # for storing RGB values for each shape. Initializing for each shape
    Reds = []
    Greens = []
    Blues = []
    
    pixels_color_counter = 0
    for pixel in shape_pixels:

         r, g, b = im_rgb_data[pixel]

         Reds.append(r)
         Greens.append(g)
         Blues.append(b)

         # maximum sample counts of each shape is 100
         if pixels_color_counter > 100:
             break
         pixels_color_counter += 1
          
    if min_colors is False:
         
         r = round(mean(Reds))
         g = round(mean(Greens))
         b = round(mean(Blues))
    else:
         # min_colors has only about 30 colors. so every shape has only one color or a few color variations. 
         # so for min_colors, get most abundant colors in the shape
         r = max(set(Reds), key = Reds.count)
         g = max(set(Greens), key = Greens.count)
         b = max(set(Blues), key = Blues.count)

    return ( r,  g,  b )





def get_all_shapes_colors(filename, directory, min_colors=False, xy_shape=False ):

   # directory is specified but does not contain /
   if directory != "" and directory[-1] != ('/'):
      directory +='/'

   if not xy_shape: 

      image_file = top_images_dir + directory + filename + '.png'

      read_image = Image.open(image_file)
      image_width, image_height = read_image.size
      im1_rgb_data = read_image.getdata()

      shapes_dfile = top_shapes_dir + directory + "shapes/" + filename + "shapes.data"
      with open (shapes_dfile, 'rb') as fp:
         # { '79999': ['79999', ... ], ... }
         # { 'shapeid': [ pixel indexes ], ... }
         im_shapes = pickle.load(fp)
      fp.close()

   else:
      
      im1_rgb_xy_data_fpath = top_shapes_dir + directory + "shapes/im_rgb_xy_data/" + filename + ".data"
      with open (im1_rgb_xy_data_fpath, 'rb') as fp:
         im1_rgb_data = pickle.load(fp)
      fp.close()

      shapes_dfile = top_shapes_dir + directory + "shapes/" + filename + "xy_shapes.data"
      with open (shapes_dfile, 'rb') as fp:
         # { '79999': ['79999', ... ], ... }
         # { 'shapeid': [ pixel indexes ], ... }
         im_shapes = pickle.load(fp)
      fp.close()


   # { shapeid: (r,g,b), ... }
   shapes_colors = {}   

   for shape_id , pixels in im_shapes.items():
      shapes_colors[shape_id] = tuple()
    
      # for storing RGB values for each shape. Initializing for each shape
      Reds = []
      Greens = []
      Blues = []
    
      pixels_color_counter = 0
      for pixel in pixels:

         r, g, b = im1_rgb_data[ pixel ]

         Reds.append(r)
         Greens.append(g)
         Blues.append(b)

         # maximum sample counts of each shape is 100
         if pixels_color_counter > 100:
             break
         pixels_color_counter += 1
          
      if min_colors is False:
         
         r = round(mean(Reds))
         g = round(mean(Greens))
         b = round(mean(Blues))
      else:
         # min_colors has only about 30 colors. so every shape has only one color or a few color variations. 
         # so for min_colors, get most abundant colors in the shape
         r = max(set(Reds), key = Reds.count)
         g = max(set(Greens), key = Greens.count)
         b = max(set(Blues), key = Blues.count)
       
      shapes_colors[shape_id] = ( r,  g,  b )

   return shapes_colors



# shapeid is the shape to get all connected shapes from shapes_list
# pass only 3 parameters. do not pass connected_shapes
def get_connected_shapes( shapeid, im_shapes_neighbors, shapes_list, connected_shapes=None ):

   for shape_nbr in im_shapes_neighbors[ shapeid ]:
      
      if shape_nbr in shapes_list:
         
         if connected_shapes is None:
            connected_shapes = set()
            connected_shapes.add( shapeid )
            connected_shapes.add( shape_nbr )
            
            get_connected_shapes( shape_nbr, im_shapes_neighbors, shapes_list, connected_shapes )
         
         elif shape_nbr in connected_shapes:
            continue
      
         elif shape_nbr not in connected_shapes:
            connected_shapes.add( shape_nbr )
            
            get_connected_shapes( shape_nbr, im_shapes_neighbors, shapes_list, connected_shapes )
            

   return connected_shapes
   


# pixels1, pixels2: set, list, or tuple of (x,y).
def get_closest_pixels( pixels1, pixels2, im_size=None, return_distance=False, return_both=False ):

   closest_distance = None
   closest_pixels = []
   
   # convert pindex to xy if provided pixels are pixel indexes.
   if type(pixels1) is set:
      if type( list(pixels1)[0] ) is int:
          pixels1 = { pixel_functions.convert_pindex_to_xy( temp_pixel, im_size[0] ) for temp_pixel in pixels1 }
          pixels2 = { pixel_functions.convert_pindex_to_xy( temp_pixel, im_size[0] ) for temp_pixel in pixels2 }
   elif type(pixels1[0]) is int:
      pixels1 = { pixel_functions.convert_pindex_to_xy( temp_pixel, im_size[0] ) for temp_pixel in pixels1 }
      pixels2 = { pixel_functions.convert_pindex_to_xy( temp_pixel, im_size[0] ) for temp_pixel in pixels2 }
   
   for pixel1 in pixels1:
      
      for pixel2 in pixels2:
         distance_x = abs( pixel1[0] - pixel2[0] )
         distance_y = abs( pixel1[1] - pixel2[1] )
         
         distance = distance_x + distance_y
         
         if closest_distance is None:
            closest_distance = distance
            closest_pixels = [ ( pixel1, pixel2 ) ]
         elif closest_distance == distance:
            closest_pixels.append( ( pixel1, pixel2 ) )
         
         elif closest_distance > distance:
            closest_distance = distance
            closest_pixels = [ ( pixel1, pixel2 ) ]
      
   if return_distance is True:
      return closest_distance
   elif return_both is True:
      return closest_distance, closest_pixels
   else:
      return closest_pixels



def combined_distance(distance1, pcout1, distance2, pcount2):
    """
    Combine two (distance, pixel_count) pairs into one.
    Returns (d_final, p_final).

    d_final = pixel-weighted average distance
    p_final = total pixel count
    """
    total_pixels = pcout1 + pcount2
    if total_pixels == 0:
        return (0, 0)  # nothing to combine

    d_final = (distance1 * pcout1 + distance2 * pcount2) / total_pixels
    return d_final, total_pixels



   

def get_pixels_w_clrs_from_shapeids( shapeids, im_shapes, im_shapes_clrs ):
    
    pixels_w_clrs = {}
    
    for shapeid in shapeids:
        
        for pixel in im_shapes[shapeid]:
            pixels_w_clrs[pixel] = im_shapes_clrs[shapeid]

    return pixels_w_clrs    

# pixels -> pixel indexes
def get_pixels_w_clrs_from_pixels( pixels, param_im_data ):
    pixels_w_clrs = {}
    for pixel in pixels:
        pixels_w_clrs[pixel] = param_im_data[pixel]
    
    return pixels_w_clrs


# see if enough pixels for the shape is contained in pixels_to_match. if so, those shapes will be matched, and their pixels
# will be returned.
# pixels_to_match is pixels with colors { pixel index: color, ... }

def get_matching_shapes_pixels( pixels_to_match, im_shapeids_by_pindex, im_shapes, im_shapes_colors, min_colors, im_size, shape_match_threshold=0.55, ignore_clrs=False, perform_match=True ):

    # { shapeid: matched count, ... }
    _matched_shapes = {}

    for pindex in pixels_to_match:
        cur_shapeids = im_shapeids_by_pindex[pindex]
        
        for cur_shapeid in cur_shapeids:
        
            if min_colors is True:
               if pixels_to_match[pindex] == tuple( im_shapes_colors[cur_shapeid] ):
                   # color is the same
                   if _matched_shapes.get(cur_shapeid) is not None:
                       _matched_shapes[cur_shapeid] += 1
                   else:
                       _matched_shapes[cur_shapeid] = 1
            else:
                appear_diff = pixel_functions.compute_appearance_difference( pixels_to_match[pindex], im_shapes_colors[cur_shapeid] )
                if appear_diff is False:
                    # same appearance
                   if _matched_shapes.get(cur_shapeid) is not None:
                       _matched_shapes[cur_shapeid] += 1
                   else:
                       _matched_shapes[cur_shapeid] = 1


    # check if shapes really match with the pixels_to_match
    match_candidate_pixels = set()
    for shapeid in _matched_shapes:
        if _matched_shapes[shapeid] / len( im_shapes[shapeid] ) >= shape_match_threshold:
            match_candidate_pixels |= set( im_shapes[shapeid] )

    if len(match_candidate_pixels) == 0:
        return set()
    
    if perform_match is False:
        return match_candidate_pixels
    
    m_candidate_shp_coord_pixels = get_shape_pos_in_shp_coord( match_candidate_pixels, im_size[0], param_shp_type=0 )
    pixels_to_match_shp_coord = get_shape_pos_in_shp_coord( pixels_to_match, im_size[0], param_shp_type=0 )
    
    match = same_shapes_functions.match_shape_while_moving_it( pixels_to_match_shp_coord, m_candidate_shp_coord_pixels, match_threshold=0.55 )
    
    if match is True:
        return match_candidate_pixels
    else:
        return set()


def get_matching_shapes_pixels2( pixels_to_match, im_rgb_data, im_shapeids_by_pindex, im_shapes, im_shapes_colors, min_colors, shape_match_threshold=0.55, ignore_clrs=False, perform_match=True ):

    # { shapeid: matched count, ... }
    _matched_shapes = {}

    for pindex in pixels_to_match:
        cur_shapeids = im_shapeids_by_pindex[pindex]
        
        for cur_shapeid in cur_shapeids:

            if min_colors is True:
               if im_rgb_data[pindex] == tuple( im_shapes_colors[cur_shapeid] ):

                   # color is the same
                   if _matched_shapes.get(cur_shapeid) is not None:
                       _matched_shapes[cur_shapeid] += 1
                   else:
                       _matched_shapes[cur_shapeid] = 1
            else:
                appear_diff = pixel_functions.compute_appearance_difference( im_rgb_data[pindex], im_shapes_colors[cur_shapeid] )
                if appear_diff is False:
                    # same appearance
                   if _matched_shapes.get(cur_shapeid) is not None:
                       _matched_shapes[cur_shapeid] += 1
                   else:
                       _matched_shapes[cur_shapeid] = 1

    # check if shapes really match with the pixels_to_match
    match_shapeids = set()
    for shapeid in _matched_shapes:
        if _matched_shapes[shapeid] / len( im_shapes[shapeid] ) >= shape_match_threshold:
            match_shapeids.add(shapeid)

    return match_shapeids



def get_surrounded_pixels( all_pixels, im_size ):
    
    surrounded_pixels = set()
    
    # convert pixel indexes to xy if provided all_pixels are pixel indexes.
    if type(all_pixels) is set:
       if type( list(all_pixels)[0] ) is int:
          all_pixels = { pixel_functions.convert_pindex_to_xy( temp_pixel, im_size[0] ) for temp_pixel in all_pixels }
   
    elif type(all_pixels[0]) is int:
       all_pixels = { pixel_functions.convert_pindex_to_xy( temp_pixel, im_size[0] ) for temp_pixel in all_pixels }


    # first, get missing virtial pixels. missing virtical pixels are missing pixels virtically where directly left and directly right pixel are both present.
    # take the very and very bottom missing virtical pixels and not the missing virtical pixels in between.
    smallest_x = min( pixel[0] for pixel in all_pixels )
    largest_x = max( pixel[0] for pixel in all_pixels )
    
    for x in range(smallest_x, largest_x + 1):
    
        cur_y_es = { pixel[1] for pixel in all_pixels if pixel[0] == x }
        
        top_y = min( temp_y for temp_y in cur_y_es  )
        bottom_y = max( temp_y for temp_y in cur_y_es )
    
        y_es_one_left = [ pixel[1] for pixel in all_pixels if pixel[0] == x - 1 ]
        y_es_one_right = [ pixel[1] for pixel in all_pixels if pixel[0] == x + 1 ]
        
        missing_top_y = None
        missing_bottom_y = None
        
        # see if upper missing pixels exist on the current column
        y_es_one_left.sort()
        for y_one_left in y_es_one_left:
            if y_one_left in y_es_one_right:
                if y_one_left < top_y:
                    # missing upper pixel found for the current column
                    all_pixels.add( (x, y_one_left) )
                break
        
        y_es_one_left.sort(reverse=True)
        for y_one_left in y_es_one_left:
            if y_one_left in y_es_one_right:
                if y_one_left > bottom_y:
                    all_pixels.add( (x, y_one_left) )
                break


    smallest_y = min( pixel[1] for pixel in all_pixels)
    largest_y = max( pixel[1] for pixel in all_pixels)

    # surrounded pixels can not be on the same top row as the surrounding pixels. so the very top surrounded pixels start at the one below the top surrounding pixels.
    # for the same reason, the very bottom surrounding pixels are one above the very bottom surrounding pixels.
    for y in range(smallest_y + 1, largest_y):

        cur_row_pixels = { pixel for pixel in all_pixels if pixel[1] == y }

        # check if further left pixels exist on 1 above and 1 below from current pixel. if so, fill in the missing pixel on current row
        x_es_one_above = [ pixel[0] for pixel in all_pixels if pixel[1] == y - 1 ]
        x_es_one_below = [ pixel[0] for pixel in all_pixels if pixel[1] == y + 1 ]
        
        x_es_one_above.sort()
        for x_one_above in x_es_one_above:
            if x_one_above in x_es_one_below:
                # left most pixel found for the missing current x where pixel 1 above and 1 below exist.
                cur_row_pixels.add( (x_one_above, y) )
                break
        
        # find missing right most pixel from both 1 above and one below
        x_es_one_above.sort(reverse=True)
        for x_one_above in x_es_one_above:
            if x_one_above in x_es_one_below:
                # rigth most pixel found
                cur_row_pixels.add( (x_one_above, y ) )
                break

        # there has to be left most and right most pixels exist to get the pixels on the current row
        if len(cur_row_pixels) < 2:
            continue
        
        # get the leftmost pixel and right most pixel at the current row.
        left_most_x = min( pixel[0] for pixel in cur_row_pixels  )
        right_most_x = max( pixel[0] for pixel in cur_row_pixels )
        
        # there has to be left most and right most pixels exist to get the pixels on the current row
        if left_most_x == right_most_x:
            continue
        
        # loop through all pixels from left_most_x to right_most_x in the current row
        for x in range( left_most_x + 1, right_most_x ): # left most surrounded pixels can be found from left most surrounding pixel + 1. same goes for right most pixel.
            
            # check if above and below pixels exist at the current x.
            pixel_above_cur_x = { pixel for pixel in all_pixels if pixel[0] == x and pixel[1] > y }
            if len(pixel_above_cur_x) == 0:
                continue
            
            pixel_below_cur_x = { pixel for pixel in all_pixels if pixel[0] == x and pixel[1] < y }
            if len(pixel_below_cur_x) == 0:
                continue
            
            surrounded_pixels.add( (x,y) )

    return surrounded_pixels



# split 2 shapes in combined shape.
def split_2shapes_in_combined_S( ovlp_shape, combined_S, im_size ):
    
    rest_comb_pixels = combined_S.difference(ovlp_shape)
    
    # get the biggest shape from rest_comb_pixels. this will be the another shape to obtain besides ovlp_shape.
    rest_comb_shapes = pixel_functions.find_shapes(rest_comb_pixels, im_size, input_xy=True )

    biggest_shape = [ set(), None ]
    for rest_comb_shapeid in rest_comb_shapes:
        if len( rest_comb_shapes[rest_comb_shapeid] ) > len(biggest_shape[0] ):
            biggest_shape[0] = rest_comb_shapes[rest_comb_shapeid]
            biggest_shape[1] = rest_comb_shapeid
    
    # biggest_shape is obtained. still there are pixels remain in rest_comb_shapes. join them to ovlp_shape or biggest_shape based on how close they are.
    for rest_comb_shapeid in rest_comb_shapes:
        if rest_comb_shapeid == biggest_shape[1]:
            continue
        
        ovlp_s_distance = get_closest_pixels( ovlp_shape, rest_comb_shapes[rest_comb_shapeid], im_size=im_size, return_distance=True )
        big_s_distance = get_closest_pixels( biggest_shape[0], rest_comb_shapes[rest_comb_shapeid], im_size=im_size, return_distance=True )

        if ovlp_s_distance < big_s_distance:
            ovlp_shape |= rest_comb_shapes[rest_comb_shapeid]

        elif ovlp_s_distance > big_s_distance:
            biggest_shape[0] |= rest_comb_shapes[rest_comb_shapeid]
        
        else:
            # both same distance.
            ovlp_shape |= rest_comb_shapes[rest_comb_shapeid]
            biggest_shape[0] |= rest_comb_shapes[rest_comb_shapeid]

    return ovlp_shape, biggest_shape[0]


# multi_s1_shapes is the list containing shapes pixels.
def split_multi_shapes_in_combined_S( multi_s1_shapes, s2_pixels, im_size ):

    ovlp_multi_s1_shapes = []
    for s1_shape in multi_s1_shapes:
    
        s1_im1pixels = s1_shape.intersection(s2_pixels)
        s2_pixels = s2_pixels.difference(s1_im1pixels)
        
        ovlp_multi_s1_shapes.append( s1_im1pixels )

    # get the biggest shape from s2_pixels. this will be the another shape to obtain besides ovlp_multi_s1_shapes.
    rest_comb_shapes = pixel_functions.find_shapes(s2_pixels, im_size, input_xy=True )

    biggest_shape = [ set(), None ]
    for rest_comb_shapeid in rest_comb_shapes:
        if len( rest_comb_shapes[rest_comb_shapeid] ) > len(biggest_shape[0] ):
            biggest_shape[0] = rest_comb_shapes[rest_comb_shapeid]
            biggest_shape[1] = rest_comb_shapeid

    #image_functions.cr_im_from_pixels( "26", "videos/street3/resized/min1/", biggest_shape[0] , pixels_rgb=(255,0,0)  )

    # biggest_shape is obtained. still there are pixels remain in rest_comb_shapes. join them to ovlp_shape or biggest_shape based on how close they are.
    all_shapes = [ biggest_shape[0] ]
    all_shapes.extend(ovlp_multi_s1_shapes )
        
    for rest_comb_shapeid in rest_comb_shapes:
        if rest_comb_shapeid == biggest_shape[1]:
            continue
        
        closest_distance = sys.maxsize      # initialize with the impossibly large distance value
        closest_shapes = set()
        
        for shape_index, shape in enumerate(all_shapes):
        
            distance = get_closest_pixels( shape, rest_comb_shapes[rest_comb_shapeid], im_size=im_size, return_distance=True )

            if distance < closest_distance:
                closest_shapes = {shape_index}
                closest_distance = distance

            elif distance == closest_distance:
                closest_shapes.add(shape_index)
                
        # join rest_comb_shape to closest shape
        for closest_shape_index in closest_shapes:
            all_shapes[closest_shape_index] |= rest_comb_shapes[rest_comb_shapeid]
        
    biggest_comb_shape = all_shapes.pop(0)
    return all_shapes, biggest_comb_shape



# split combined shape.
def split_comb_shape( shape_pixels, split_size, image_size, boundary_pixels=None ):

    if boundary_pixels is None:
        boundary_pixels = get_boundary_pixels(shape_pixels)
    
    boundary_nbrs = set() # boundary neighbor pixels
    for boundary_pixel in boundary_pixels:
        boundary_nbrs |= pixel_functions.get_nbr_pixels( boundary_pixel, image_size, input_xy=True)

    in_pixels_bnd_nbrs = boundary_nbrs.intersection(shape_pixels)
    pixel_lookup = {}
    for boundary_nbr in in_pixels_bnd_nbrs:
        pixel_lookup[boundary_nbr] = True

    #image_functions.cr_im_from_pixels( image2_fname, directory, boundary_pixels, pixels_rgb=( 0, 0, 255 ) ) # (255, 100, 80 ) orange

    maybe_1p_conns = set()
    for boundary_p in boundary_pixels:
        # separation patterns. when pixels are missing on both opposite sides. ( left, right ). ( top, bottom ). ( left up, right down ). ( right up, left down )
        left =  ( boundary_p[0] - 1, boundary_p[1] )
        right = ( boundary_p[0] + 1, boundary_p[1] )
        top =   ( boundary_p[0], boundary_p[1] - 1 )
        bottom = ( boundary_p[0], boundary_p[1] + 1 )
        left_up = ( boundary_p[0] - 1, boundary_p[1] - 1)
        right_down = ( boundary_p[0] + 1, boundary_p[1] + 1)
        right_up = ( boundary_p[0] + 1, boundary_p[1] - 1)
        left_down = ( boundary_p[0] - 1, boundary_p[1] + 1)
        
        nbr_pixels = pixel_functions.get_nbr_pixels( boundary_p, image_size, input_xy=True)
        nbr_pixels = { temp_pixel for temp_pixel in nbr_pixels if pixel_lookup.get(temp_pixel) is not None } # get neighbor pixels of the current boundary_p but remove pixels outside the shape_pixels.
        
        if left not in nbr_pixels and right not in nbr_pixels:
            maybe_1p_conns.add(boundary_p)
        elif top not in nbr_pixels and bottom not in nbr_pixels:
            maybe_1p_conns.add(boundary_p)
        elif left_up not in nbr_pixels and right_down not in nbr_pixels:
            maybe_1p_conns.add(boundary_p)
        elif right_up not in nbr_pixels and left_down not in nbr_pixels:
            maybe_1p_conns.add(boundary_p)
    
    # now separate maybe_1p_conns by connected pixels. in 1 connected shape from maybe_1p_conns, if cutting 1 pixel makes the nearby shape_pixels unreachable, then this further indicates the connection point
    maybe_1p_shapes = pixel_functions.find_shapes(maybe_1p_conns, image_size, input_xy=True )
    maybe_conn_points = []
    
    for maybe_1p_shapeid in maybe_1p_shapes:
        all_nbearby_pixels = set()
        for pixel in maybe_1p_shapes[maybe_1p_shapeid]:
            nbr_pixels = pixel_functions.get_nbr_pixels( pixel, image_size, input_xy=True)
            all_nbearby_pixels |= nbr_pixels.intersection( in_pixels_bnd_nbrs )
        
        if len( maybe_1p_shapes[maybe_1p_shapeid] ) == 1:
            all_nbearby_pixels |= maybe_1p_shapes[maybe_1p_shapeid]
        
        conn_shape = pixel_functions.find_shapes(all_nbearby_pixels, image_size, input_xy=True ) 
        assert len(conn_shape) == 1 # this should be one connected pixels

        for pixel in list( conn_shape.values() )[0]:
            # check if removing the current pixel separates the current conn_shape. if so, it is maybe connection point
            
            one_conn_pixels = set( list( conn_shape.values() )[0] )
            one_conn_pixels.remove(pixel)
            
            after_rm_shape = pixel_functions.find_shapes(one_conn_pixels, image_size, input_xy=True )
            if len(after_rm_shape) >= 2:
                maybe_conn_points.append( maybe_1p_shapes[maybe_1p_shapeid] )
                break
        
    '''
    image_functions.cr_im_from_pixels( image2_fname, directory, maybe_1p_conns, pixels_rgb=( 0, 0, 255 ) )
    for pixels_index, pixels in enumerate(maybe_conn_points):
        if not os.path.exists(test_imfpath):
            image_functions.cr_im_from_pixels( image2_fname, directory, pixels, pixels_rgb=( 255, 0, 255 ), save_filepath=test_imfpath )
        elif pixels_index < len(maybe_conn_points) - 1:
            image_functions.cr_im_from_pixels( image2_fname, directory, pixels, pixels_rgb=( 255, 0, 255 ), save_filepath=test_imfpath, input_im=test_imfpath )
        
        else:
            # last one
            image_functions.cr_im_from_pixels( image2_fname, directory, pixels, pixels_rgb=( 255, 0, 255 ), input_im=test_imfpath )
    '''

    shape_inner_pixels = shape_pixels.difference(boundary_pixels)

    # to get real connection point. for each connection point, get neighbor pixels from all pixels of the connection point. if neighbor pixels are all connected together, 
    # this will not split the target whole shape. if neighbor pixels are separate, then check if removing the connection point pixels split the whole shape
    separated_shapes = []
    for conn_point in maybe_conn_points:
        
        nbr_pixels = set()
        for pixel in conn_point:
            nbr_pixels |= pixel_functions.get_nbr_pixels( pixel, image_size, input_xy=True)

        in_nbr_pixels = nbr_pixels.intersection(in_pixels_bnd_nbrs) # neighbor pixels inside the shape.
        in_nbr_pixels = in_nbr_pixels.difference(conn_point)
        
        nbr_shapes = pixel_functions.find_shapes(in_nbr_pixels, image_size, input_xy=True )
        if len(nbr_shapes) == 1:
            # all neighbor pixels are connected. no split here
            continue
        
        tshape_pixels_cp = shape_pixels.copy()
        tshape_pixels_cp = tshape_pixels_cp.difference(conn_point)
        
        split_shapes = pixel_functions.find_shapes(tshape_pixels_cp, image_size, input_xy=True )
        if len(split_shapes) == 1:
            continue # no split
        
        split_fail = False
        for split_pixels in split_shapes.values():
            if len(split_pixels) < split_size:
                split_fail = True
                break
        if split_fail is True:
            continue
        
        separated_shapes.append( ( conn_point, list( split_shapes.values() ) ) )
        
        '''
        image_functions.cr_im_from_pixels( image2_fname, directory, conn_point, pixels_rgb=( 0, 255, 255 ), save_filepath=test_imfpath )
        for split_index, split_pixels in enumerate( split_shapes.values() ):
            
            if split_index < len( split_shapes.values() ) - 1:
                image_functions.cr_im_from_pixels( image2_fname, directory, split_pixels, pixels_rgb=( 0, 0, 255 ), save_filepath=test_imfpath, input_im=test_imfpath )
            else:
                # last
                image_functions.cr_im_from_pixels( image2_fname, directory, split_pixels, pixels_rgb=( 0, 0, 255 ), input_im=test_imfpath )
        '''


    all_split_pixels = []
    for each_sep in separated_shapes:
        # ( connection point pixels, [ split pixels, ... ] )
        for split_pixels in each_sep[1]:
            split_pixels |= each_sep[0]
            
            all_split_pixels.append( split_pixels )
            #image_functions.cr_im_from_pixels( image2_fname, directory, split_pixels, pixels_rgb=( 0, 255, 255 ) )


    done_splits = set()
    ovlp_threshold = 0.95
    for split_index, split_pixels in enumerate(all_split_pixels):
        for ano_split_index, ano_split_pixels in enumerate(all_split_pixels):
            if split_index == ano_split_index:
                continue
            if ( split_index, ano_split_index ) in done_splits or ( ano_split_index, split_index ) in done_splits:
                continue
            
            done_splits.add( (split_index, ano_split_index) )
            
            overlap_pixels = split_pixels.intersection(ano_split_pixels)
            ovlp_percent = len(overlap_pixels) / len(split_pixels)
            ano_ovlp_percent = len(overlap_pixels) / len(ano_split_pixels)
            if ovlp_percent >= ovlp_threshold and ano_ovlp_percent <  ovlp_threshold:
                # most of split_pixels contained in ano_split_pixels. delete split_pixels from ano_split_pixels
                ano_split_p_after = ano_split_pixels.difference(split_pixels)
                all_split_pixels[ano_split_index] = ano_split_p_after
            
            elif ano_ovlp_percent >=  ovlp_threshold and ovlp_percent <  ovlp_threshold:
                split_p_after = split_pixels.difference(ano_split_pixels)
                all_split_pixels[split_index] = split_p_after
            


    # delete almost duplicates
    duplicates = set()
    done_splits = set()
    for split_index, split_pixels in enumerate(all_split_pixels):
        for ano_split_index, ano_split_pixels in enumerate(all_split_pixels):
            if split_index == ano_split_index:
                continue
            if ( split_index, ano_split_index ) in done_splits or ( ano_split_index, split_index ) in done_splits:
                continue

            done_splits.add( (split_index, ano_split_index) )
            
            overlap_pixels = split_pixels.intersection(ano_split_pixels)
            ovlp_percent = len(overlap_pixels) / len(split_pixels)
            ano_ovlp_percent = len(overlap_pixels) / len(ano_split_pixels)
            if ovlp_percent >= ovlp_threshold and ano_ovlp_percent >= ovlp_threshold:
                # duplicate. delete either one
                duplicates.add(split_index)

    duplicates = list(duplicates)
    duplicates.sort()

    deleted = 0
    for duplicate_index in duplicates:
        all_split_pixels.pop( duplicate_index - deleted )
        deleted += 1

    return all_split_pixels






# return image1 to image2 movement and overlap pixels at that movement
def get_movement(image1pixels, image2pixels):
    counts = Counter()
    
    # Step 1. Count all movement vectors
    for x1, y1 in image1pixels:
        for x2, y2 in image2pixels:
            dx = x2 - x1
            dy = y2 - y1
            counts[(dx, dy)] += 1

    # Step 2. Find the most frequent movement vector
    best_dx, best_dy = max(counts, key=counts.get)

    # Step 3. Compute overlap pixels for that best movement
    moved_image1 = {(x + best_dx, y + best_dy) for (x, y) in image1pixels}
    overlap_pixels = moved_image1.intersection(image2pixels)

    return (best_dx, best_dy), overlap_pixels



def get_mv_with_buffer( movement, buffer ):

    dx, dy = movement
    expanded_set = {(x, y) for x in range(dx - buffer, dx + buffer + 1)
                             for y in range(dy - buffer, dy + buffer + 1)}

    return expanded_set











