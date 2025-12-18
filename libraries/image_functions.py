from libraries.cv_globals import top_shapes_dir, top_images_dir
from libraries import pixel_functions
from PIL import Image
import os, sys
import math
import re
import pickle

from libraries import cv_globals



def get_image_areas( im_file, directory ):

   # needed for creating image areas
   image = Image.open(top_images_dir + directory + im_file + ".png")
   im_width, im_height = image.size
   
   # image width can not have decimal point values because decimal point value creates number less than the actual image width or height.
   # so divider has to be the one that can exactly divide image width or height
   
   # divide image into columns and rows
   image_dividers = [ 5, 6, 7, 8, 9, 4, 10, 11, 12, 13 ]
   
   image_divider = None
   for temp_divider in image_dividers:
      if im_width % temp_divider == 0 and im_height % temp_divider == 0:
         
         im_area_width = round(im_width / temp_divider)
         im_area_height = round(im_height / temp_divider)
         
         image_divider = temp_divider
         
         break
   
   if not image_divider:
      print("ERROR at get_image_areas method. No image divider found")
      sys.exit(1)
   
   image_areas = {}
   
   column_num = 0
   for column_num in range(0, image_divider):
      for row_num in range(0, image_divider):
         if row_num == 0:
            left = 0
            right = im_area_width
         else:
            left = (im_area_width * row_num) + 1
            right = (im_area_width * row_num) + im_area_width
         
         if column_num == 0:
            top = 0
            bottom = im_area_height
         else:
            top = ( column_num * im_area_height ) + 1
            bottom = ( column_num * im_area_height ) + im_area_height
         
         image_area = row_num + 1 + ( column_num * image_divider )
         image_areas[image_area ] = ( left, right, top, bottom )

   return image_areas




# this function is to get following image data
# counts of shapes in the image
# how many shapes there are in the image that consist of small numbers of pixels

# shapes_type
# normal is the one created by find_shapes
# intnl_spixcShp are shapes that incorporate internal small pixel count shapes
def get_image_stats(imfile, directory, shapes_type=None):

   imshapes = None

   shapes_dir = top_shapes_dir + directory + "shapes/"   
   shapes_dfile = shapes_dir + imfile + "shapes.data"

   with open (shapes_dfile, 'rb') as fp:
      # { '79999': ['79999', ... ], ... }
      # { 'shapeid': [ pixel indexes ], ... }
      imshapes = pickle.load(fp)
   fp.close() 


   imshapes_counts = len( imshapes )
   
   #print("image shapes counts: " + str( imshapes_counts ) )
   
   # contains each shape's pixel counts
   # {'shapeid': pixel count, 'shapeid': pixel count, .... }
   shapes_counts = {}

   for shapeid in imshapes:
      shapes_counts[shapeid] = len( imshapes[shapeid] )

   
   # creating pixel count statistics
   # { pixel count : counts of shapes that have this pixel count, pixel count: counts of shapes that have this pixel count, ... }
   pixel_counts = {}
   
   pixel_counts_w_shapes = { }
   for shapeid in shapes_counts:
      cur_pcount = shapes_counts[shapeid]
      
      
      
      if len( pixel_counts ) == 0:
         pixel_counts[ cur_pcount ] = 1
         
         pixel_counts_w_shapes[ cur_pcount ] = [ shapeid ]
   
      elif cur_pcount in pixel_counts.keys():
         #cur_pcount is found in pixel_counts. add 1
         pixel_counts[ cur_pcount ]  += 1
         
         pixel_counts_w_shapes[ cur_pcount ].append( shapeid )
      
      else:
         #cur_pcount is not found in pixel_counts. create this pixel count
         pixel_counts[ cur_pcount ] = 1

         pixel_counts_w_shapes[ cur_pcount ] = [ shapeid ]
   
   pixel_counts = dict( sorted(pixel_counts.items(), key=lambda i: i[1], reverse= True) )
   #print( pixel_counts )

   return pixel_counts_w_shapes



# this function incorporates internal small pixel count shapes and reads shapes file from 
# top_shapes_dir/videos/video_filename/image_directory/shapes/intnl_s_pixcShp/data
# shapes_rgbs -> list or tuple of (r,g,b).
def cr_im_from_shapeslist2( imfname, imdir, in_shapes, save_filepath=None , shapes_rgbs=None, input_im=None ):
   
   if input_im is not None:
      original_image = Image.open( input_im )
   else:
      original_image = Image.open(top_images_dir + imdir + imfname + ".png")
   
   image_width, image_height = original_image.size

   
   if shapes_rgbs is None:
      shapes_colors =  get_rgbs(100)

   
   # read shapes file
   shapes_dfile = top_shapes_dir + imdir + "shapes/" + imfname + "shapes.data"

   with open (shapes_dfile, 'rb') as fp:
      # { 'shapeid': [ pixel indexes ], ... }
      im_shapes_w_intl_spixcShp = pickle.load(fp)
   fp.close()

   rgb_color_counter = 0
   for shapeid in in_shapes:
      pixch_shape_pixels = im_shapes_w_intl_spixcShp[ shapeid ]

      for pixel in pixch_shape_pixels:
   
         y = math.floor( pixel / image_width)
         x  = pixel % image_width
         
         if shapes_rgbs is not None:
            if rgb_color_counter == len(shapes_rgbs):
               rgb_color_counter = 0
            original_image.putpixel( (x , y) , shapes_rgbs[rgb_color_counter] )
         elif shapes_rgbs is None:
            if len( shapes_colors ) <= rgb_color_counter:
               rgb_color_counter = 0
               
            original_image.putpixel( (x , y) , shapes_colors[ rgb_color_counter ] )
      
      rgb_color_counter += 1

   if save_filepath is not None:
      original_image.save( save_filepath )
   elif save_filepath is None:
      original_image.show()

   original_image.close()


# pixels -> [ indexes ] or [ ( x, y ), ... ] when with_colors is False
# pixels -> { pixel index: (r,g,b), ... } when with_colors is True
#
# pixels_rgb -> ( r, g, b )  r,g,b are integers
def cr_im_from_pixels( imfname, imdir, pixels, save_filepath=None , pixels_rgb=None, input_im=False, with_colors=False ):

   if cv_globals.debug is None or cv_globals.debug is False:
      return

   if input_im is False:
      imfname = imfname.replace(".png", '')
      if "output" not in imdir:
         original_image = Image.open(top_images_dir + imdir + imfname + ".png")
      else:
         original_image = Image.open(imdir + imfname + ".png")
        
   else:
      original_image = Image.open(input_im)

   image_width, image_height = original_image.size
   
   if pixels_rgb is None and with_colors is False:
      pixels_rgb = (255,0,0)

   if with_colors is False:
      if type( list(pixels)[0] ) is int:
         for pixel in pixels:   
            y = math.floor( pixel / image_width)
            x  = pixel % image_width
            
            if y < 0 or x < 0 or y >= image_height or x >= image_width:
               continue

            original_image.putpixel( (x , y) , pixels_rgb )
            
      elif type( list(pixels)[0] ) is tuple or type( list(pixels)[0] ) is list:
         for pixel in pixels:
            if pixel[1] < 0 or pixel[0] < 0 or pixel[1] >= image_height or pixel[0] >= image_width:
               continue

            original_image.putpixel( pixel, pixels_rgb )       
            

   elif with_colors is True:
      # pixels -> { pixel index: (r,g,b), ... }
      
      for pixel in pixels:
         y = math.floor( pixel / image_width)
         x  = int(pixel) % image_width  

         if y < 0 or x < 0 or y >= image_height or x >= image_width:
            continue         
         original_image.putpixel( (x,y), pixels[pixel] )


   if save_filepath is not None:
      original_image.save( save_filepath )
   elif save_filepath is None:
      original_image.show( )


# pixels can be pixel indexes or xy pixels.
# 
def show_pixels_with_mono_color_background( pixels, im_data, im_size, color_name ):

    image_obj = Image.new("RGB", im_size, color_name )
    im_pixels = image_obj.load()

    pixels = list(pixels)
    if type( list(pixels)[0] ) is int:
        # pixel indexes are in pixels.
        for pindex in pixels:
            xy = pixel_functions.convert_pindex_to_xy( pindex, im_size[0] )
            
            im_pixels[ xy[0], xy[1] ] = im_data[pindex]

    else:
        for xy in pixels:
            pindex = pixel_functions.convert_xy_to_pindex( xy, im_size[0] )
            
            im_pixels[ xy[0], xy[1] ] = im_data[pindex]
    
    image_obj.show()
        
        
    








# gets color that is more than threshold away from every color in color_variations.
# color_variations are from top_shapes_dir + directory + "data/color_variations.data"
def  get_non_existing_colors( color_variations, default_clr_threshold, rgb_steps=50 ):
   
   
   all_rgbs = get_rgbs( steps= int(rgb_steps) )
   
   existing_colors = set()
   for color_variation in color_variations:
      for pos_negative1 in [ 1, -1 ]:
         for threshold_r in range( default_clr_threshold, 0 - 1, -1 ):
            threshold_r *= pos_negative1
            
            r = color_variation[0] - threshold_r
            
            for pos_negative2 in [ 1, -1 ]:
               
               for threshold_g in range( default_clr_threshold, 0 - 1, -1 ):
                  threshold_g *= pos_negative2
                  
                  g = color_variation[1] - threshold_g
                  
                  for pos_negative3 in [ 1, -1 ]:
                     
                     for threshold_b in range( default_clr_threshold, 0 - 1, -1 ):
                        threshold_b *= pos_negative3
                        
                        b = color_variation[2] - threshold_b
                        
                        if r < 0 or g < 0 or b < 0 or r > 255 or g > 255 or b > 255:
                           # not rgb value
                           continue
                        
                        existing_colors.add( (r,g,b) )
                        

   non_existing_colors = set(all_rgbs).difference( existing_colors )
   
   return non_existing_colors


# this can be used to pass as a parameter to get_non_existing_colors
def get_rgb_step( color_variations_len ):
   # for every 10 increase in color_variations_len, rgb_steps decrease by 8
   # reference value is when color_variations_len is 16 with rgb_steps of 50
   clr_variations_ref = 16
   rgb_steps_ref = 50
   
   clr_variations_diff_from_ref = ( color_variations_len - clr_variations_ref ) / 10 
   
   rgb_steps = rgb_steps_ref - ( clr_variations_diff_from_ref * 8 ) 
   
   return rgb_steps
      



# original_image_data is from pillow image object getdata() or list of rgb values.
# move_RL and move_UD have the same form. negative value for left and up. positive value for right and down
# non_image_color is the one returned by image_functions.get_non_existing_colors but only 1 pixel. example. (255, 30, 20 ) 

# return { pindex: ( r,g,b ), ... }
def move_image( original_image_data, move_RL, move_UD, image_size, non_image_color ):
   last_image_index = ( image_size[0] * image_size[1] ) - 1

   moved_image = {}
   
   if move_RL > 0:
      # image moved right, therefore non image starts from x = 0.
      # put all non image colors for leftside of x.
      for x in range( move_RL ):
         for y in range( image_size[1] ):
            pindex = pixel_functions.convert_xy_to_pindex( (x,y), image_size[0] )
            moved_image[pindex] = non_image_color

   else:
      # image moved left, therefore non image starts from x = image width - 1
      # put all non image colors for rightside of x.
      start_non_im_x = image_size[0] + move_RL
      for x in range( start_non_im_x, image_size[0] ):
         for y in range( image_size[1] ):
            pindex = pixel_functions.convert_xy_to_pindex( (x,y), image_size[0] )
            moved_image[pindex] = non_image_color   

   
   if move_UD > 0:
      # move image down. therefore non image starts from y = 0
      # put all non image colors for the upper side of y
      for y in range( move_UD ):
         for x in range( image_size[0] ):      
            pindex = pixel_functions.convert_xy_to_pindex( (x,y), image_size[0] )
            moved_image[pindex] = non_image_color
   
   else:
      # move image up. therefore non image starts from y = image_height - 1
      start_non_im_y = image_size[1] + move_UD
      for y in range( start_non_im_y, image_size[1] ):
         for x in range( image_size[0] ):
            pindex = pixel_functions.convert_xy_to_pindex( (x,y), image_size[0] )
            moved_image[pindex] = non_image_color
   
      
 
   for image_index in range( 0, last_image_index + 1 ):
      y = math.floor( image_index / image_size[0])
      x  = image_index % image_size[0]       
      
      moved_x = x + move_RL
      
      if moved_x < 0 or moved_x >= image_size[0] :
         # moved out of the image range.
         continue
      
      moved_y = y + move_UD
      if moved_y >= image_size[1] or moved_y < 0:
         continue
      
      moved_pindex = pixel_functions.convert_xy_to_pindex( (moved_x,moved_y), image_size[0] )
      
      moved_image[moved_pindex] = original_image_data[image_index]   

   return moved_image






def get_rgbs( start_r=0, start_g=0, start_b=0, steps=50 ):
   rgbs = []
   
   def _get_rgbs( _start_r, _start_g, _start_b, end_r=255, end_g=255, end_b=255, _steps=50 ):
   
      for r in range( start_r, end_r, _steps ):
         for g in range( start_g, end_g, _steps ):
            for b in range( start_b, end_b, _steps ):            
               rgbs.append( (r,g,b) )   
   
   
   _get_rgbs( start_r, start_g, start_b, _steps=steps )
   
   _get_rgbs( 0,0,0, end_r=start_r + 1, end_g=start_g + 1, end_b=start_b + 1, _steps=steps )
   
   return rgbs      












































