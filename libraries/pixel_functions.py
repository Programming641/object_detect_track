
import sys
import math
import copy
from libraries import image_functions, pixel_shapes_functions
from collections import deque


# return True if appearance changed, False if appearance did not change
def compute_appearance_difference(orig_pix, comp_pix, threshold=20 ):
   
   assert type(orig_pix) is tuple and type(comp_pix) is tuple
   
   red_difference = abs( orig_pix[0] - comp_pix[0] )
   green_difference = abs( orig_pix[1] - comp_pix[1] )
   blue_difference = abs( orig_pix[2] - comp_pix[2] )    

   total_diff = red_difference + green_difference + blue_difference
   if total_diff > threshold:
      return True
   else:
      return False



# xy -> ( x, y ) x and y are integers
# return pixel_index. pixel_index is integer
def convert_xy_to_pindex( xy, im_width ):
   pixel_index = xy[1] * im_width + xy[0]
   
   return pixel_index

def convert_pindex_to_xy( pindex, im_width ):
   y = math.floor( pindex / im_width)
   x  = pindex % im_width 

   return ( x,y )


def convert_pindexes_to_xy( pindexes, im_size ):
    xy_pixels = set()
    for pindex in pindexes:
        y = math.floor( pindex / im_size[0])
        x  = pindex % im_size[0] 

        xy_pixels.add( (x,y) )

    return xy_pixels


def convert_xy_pixels_to_pindexes( xy_pixels, im_size ):
    pindexes = set()
    
    for xy_pixel in xy_pixels:
        pindex = convert_xy_to_pindex( xy_pixel, im_size[0] )
        pindexes.add(pindex)
    
    return pindexes
    


# out of image pixel can not be obtained with pixel index
def get_pindex_with_xy_input( pindex, param_xy, im_size ):

    param_pindex_xy = convert_pindex_to_xy( pindex, im_size[0] )   
    x = param_pindex_xy[0] + param_xy[0]
    y = param_pindex_xy[1] + param_xy[1]
    
    result_pindex = pindex + param_xy[0] + ( param_xy[1] * im_size[0] )
    
    if x < 0 or x >= im_size[0] or y < 0 or y >= im_size[1]:
        # out of image range
        return None

    return result_pindex



def get_nbr_pixels( pixel, image_size, input_xy=False):
    """
    Return list of neighboring pixel coordinates (including diagonals).
    
    (x, y) -> target pixel
    width, height -> image dimensions
    """
    if not input_xy:
        xy = convert_pindex_to_xy( pixel, image_size[0] )
    else:
        xy = pixel
    
    x = xy[0]
    y = xy[1]
    
    width = image_size[0]
    height = image_size[1]
    
    neighbors = set()
    # directions: 8 neighbors (dx, dy)
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1),  (1, 0), (1, 1)
    ]
    
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:  # stay inside bounds
            if not input_xy:
                neighbor = convert_xy_to_pindex( (nx, ny), width )
            else:
                neighbor = (nx, ny)
            
            neighbors.add(neighbor)
    
    return neighbors



# no color checking
def find_shapes(pixels, image_size, input_xy=None ):

    visited = set()
    pixel_lookup = {}
    for pixel in pixels:
        pixel_lookup[pixel] = True

    shapes = {}

    for pixel in pixels:
        if pixel in visited:
            continue

        queue = deque([pixel])
        shape_pixels = set()

        while queue:
            current = queue.popleft()
            if current in visited or not pixel_lookup.get(pixel):
                continue

            visited.add(current)
            shape_pixels.add(current)

            for nbr in get_nbr_pixels(current, image_size, input_xy=input_xy):
                if nbr not in visited and pixel_lookup.get(nbr):
                    queue.append(nbr)

        if shape_pixels:
            shape_id = next(iter(shape_pixels))  # Any pixel from the shape
            shapes[shape_id] = shape_pixels

    return shapes



# find_shapes does not perform color checking. this one does.
def find_shapes2(pixels, image_rgbs, image_size, min_color, input_xy=False):

    visited = set()
    
    pixel_lookup = {}
    for pixel in pixels:
        pixel_lookup[pixel] = True
    
    shapes = {}

    for pixel in pixels:
        if pixel in visited:
            continue

        color = image_rgbs[pixel]
        queue = deque([pixel])
        shape_pixels = set()

        while queue:
            current = queue.popleft()
            if current in visited or not pixel_lookup.get(current):
                continue
            
            if min_color:
                if image_rgbs[current] != color:
                    continue
            else:
                clr_diff = compute_appearance_difference(image_rgbs[current], color )
                if clr_diff:
                    continue

            visited.add(current)
            shape_pixels.add(current)

            for nbr in get_nbr_pixels(current, image_size, input_xy=input_xy):
                if nbr not in visited and pixel_lookup.get(nbr):
                    if min_color:
                        if image_rgbs[nbr] == color:
                            queue.append(nbr)
                    
                    else:
                        clr_diff = compute_appearance_difference(image_rgbs[nbr], color )
                        if not clr_diff:
                            queue.append(nbr)


        if shape_pixels:
            shape_id = next(iter(shape_pixels))  # Any pixel from the shape
            shapes[shape_id] = shape_pixels

    return shapes


# 
# get same color connected pixels. the connected pixels don't have to be inside the provided parameter pixels. just traverse the same color pixels in the image.
def get_same_clr_conn_pixels(pixels, image_rgbs, image_size, min_colors, input_xy=False):

    visited = set()
    shapes = {}

    for pixel in pixels:
        if pixel in visited:
            continue

        color = image_rgbs[pixel]
        queue = deque([pixel])
        shape_pixels = set()

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            
            if min_colors:
                if image_rgbs[current] != color:
                    continue
            else:
                clr_diff = compute_appearance_difference(image_rgbs[current], color )
                if clr_diff:
                    continue

            visited.add(current)
            shape_pixels.add(current)

            for nbr in get_nbr_pixels(current, image_size, input_xy=input_xy):
                if nbr not in visited:
                    if min_colors:
                        if image_rgbs[nbr] == color:
                            queue.append(nbr)
                    
                    else:
                        clr_diff = compute_appearance_difference(image_rgbs[nbr], color )
                        if not clr_diff:
                            queue.append(nbr)


        if shape_pixels:
            shape_id = next(iter(shape_pixels))  # Any pixel from the shape
            shapes[shape_id] = shape_pixels

    return shapes




def expand_boundary(start_pixels, image_size, expand_by=1, input_xy=False):
    expanded_set = set(start_pixels)
    
    boundary_pixels = set( pixel_shapes_functions.get_boundary_pixels(start_pixels, im_size=image_size) )

    for _ in range(expand_by):
        new_pixels = set()
        
        for pixel in boundary_pixels:
            neighbors = get_nbr_pixels(pixel, image_size, input_xy=input_xy)
            new_pixels.update(neighbors)
        boundary_pixels.update(new_pixels)
    
    expanded_set.update(boundary_pixels)
    
    # remove pixels that went out of image.
    rev_expanded_set = set()
    for expanded_pixel in expanded_set:
        if expanded_pixel[0] < 0 or expanded_pixel[0] >= image_size[0] or expanded_pixel[1] < 0 or expanded_pixel[1] >= image_size[1]:
            continue
        rev_expanded_set.add(expanded_pixel)
    
    return rev_expanded_set




# get pixels positions in pixel coordinate and not image coordinate
#
# pixels -> { (x,y), ... } or list of xy
# 
# shapes_colors is the one returned by get_all_shapes_colors
# 
# returns list of (x,y).   [ ( x,y ), ... ]
# x,y and r,g,b are all integers
def get_pixels_pos_in_pix_coord( pixels ): 
      
   smallest_x = min( [ xy[0] for xy in pixels ] )
   smallest_y = min( [ xy[1] for xy in pixels ] )

   pixel_coords = [ (xy[0] - smallest_x, xy[1] - smallest_y ) for xy in pixels ]

   return pixel_coords


# pixels -> [ (x,y), .... ] or set of xy for xy pixels. it can also be pixel indexes. 
# im_size is required.
# move_x or move_y.   postive value. 1,2,3,4.... move right or down 1,2,3,4..... negative value -> -1,-2,-3,... move left or up 1,2,3,...
# out of image pixels can not be obtained with pixel index.
def move_pixels( pixels, move_x, move_y, input_xy=True, im_size=None, get_out_of_im=False ):
   moved_pixels = set()
   out_of_image_pixels = set()
   
   if get_out_of_im is True:
       # out of image pixel can only be obtained with xy vaues and not pixel index.
       if input_xy is False:
           pixels = convert_pindexes_to_xy( pixels, im_size )
           input_xy = True
   
   if input_xy is True:
      for pixel in pixels:
         x = pixel[0] + move_x
         y = pixel[1] + move_y
         if x >= im_size[0] or x < 0 or y >= im_size[1] or y < 0:
             # out of image range
             out_of_image_pixels.add( (x,y) )
             continue
         moved_pixels.add( (x,y) )
   
   else:
      for pindex in pixels:
         result_pindex = get_pindex_with_xy_input( pindex, (move_x, move_y), im_size )
         if result_pindex is None:
            # out of image range
            continue

         moved_pixels.add( result_pindex )
   
   if get_out_of_im is False:
      return moved_pixels
   else:
      return moved_pixels, out_of_image_pixels



# target_pixels are the ones to find shapes from. all target_pixels are inside the whole_pixels
def get_target_shapes( target_pixels, whole_pixels, image_size, input_xy=True ):
   
    # to speed up the lookup time, whole_pixels will be put in keys.
    whole_p_lookup = {}
    for whole_pixel in whole_pixels:
        whole_p_lookup[whole_pixel] = True # value can be anything. just need to put whole_pixel in key for lookup
    
    shapes = {}
    done_pixels = {} # put done pixel in dictionary key for instant time lookup
    
    '''
    def _traverse_all_neighbors( shapeid, param_pixel ):

        nbr_pixels = get_nbr_pixels(param_pixel, image_size, input_xy=input_xy)
        for nbr_pixel in nbr_pixels:
            if done_pixels.get(nbr_pixel) or not whole_p_lookup.get(nbr_pixel):
                # nbr_pixel is already done or is not in the whole_pixels
                continue
            shapes[shapeid].add(nbr_pixel)
            done_pixels[nbr_pixel] = True
            
            _traverse_all_neighbors( shapeid, nbr_pixel )   
    '''

    def _traverse_all_neighbors_iter(shapeid, start_pixel):
        stack = [start_pixel]
        while stack:
            pixel = stack.pop()
            nbr_pixels = get_nbr_pixels(pixel, image_size, input_xy=input_xy)
            for nbr_pixel in nbr_pixels:
                if done_pixels.get(nbr_pixel) or not whole_p_lookup.get(nbr_pixel):
                    continue
                shapes[shapeid].add(nbr_pixel)
                done_pixels[nbr_pixel] = True
                stack.append(nbr_pixel)
    
    
    for t_pixel in target_pixels:
        if done_pixels.get(t_pixel): # this pixel has already been put in shape.
            continue
        
        shapes[t_pixel] = {t_pixel}
        done_pixels[t_pixel] = True
        
        t_nbr_pixels = get_nbr_pixels(t_pixel, image_size, input_xy=input_xy)
        for t_nbr_pixel in t_nbr_pixels:
            if done_pixels.get(t_nbr_pixel) or not whole_p_lookup.get(t_nbr_pixel):
                # t_nbr_pixel is already done or is not in the whole_pixels
                continue
            
            shapes[t_pixel].add(t_nbr_pixel)
            done_pixels[t_nbr_pixel] = True
            
            _traverse_all_neighbors_iter( t_pixel, t_nbr_pixel )

          
    return shapes



def  get_same_clr_pixels( pixels, from_im_rgb_data, to_match_im_rgb_data, min_colors, same_clr=True):

	matched_pixels = set()
	for pixel in pixels:
		if min_colors is True:
			if to_match_im_rgb_data[pixel] == from_im_rgb_data[pixel]:
				matched_pixels.add(pixel)
		
		else:
			appear_diff = compute_appearance_difference( from_im_rgb_data[pixel], to_match_im_rgb_data[pixel] )
			if appear_diff is False:
				# same appearance
				matched_pixels.add(pixel)
	
	if same_clr is True:
		return matched_pixels
	else:
		non_m_pixels = pixels.difference(matched_pixels)
		return non_m_pixels

































