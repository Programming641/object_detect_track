from libraries.cv_globals import proj_dir, top_shapes_dir, top_images_dir
from libraries import pixel_shapes_functions, pixel_functions
from PIL import Image
import os, sys
import math
import re
import pickle


# btwn_files are the image frames between files such as [ 25.26, 26.27, ... ] that need to be ordered
# all_btwn_files_nums are the return values of btwn_amng_files_functions.get_btwn_files_nums()
def get_files_in_order( btwn_files, all_btwn_files_nums ):

   ordered_btwn_files = []
   
   for btwn_files_iter in all_btwn_files_nums:
      if btwn_files_iter in btwn_files:
         ordered_btwn_files.append(btwn_files_iter)

   return ordered_btwn_files



def get_btwn_files_nums( lowest_filenum, highest_filenum ):
   all_btwn_files_nums = []
   for cur_num in range( int( lowest_filenum ), int( highest_filenum ) ):
      im1_filenum = str( cur_num )
      im2_filenum = str( cur_num + 1 )

      all_btwn_files_nums.append( im1_filenum + "." + im2_filenum )

   return all_btwn_files_nums


# imdir is directory under top_images_dir
def get_low_highest_filenums(imdir):

   imdir_contents = os.listdir(top_images_dir + imdir)
   im_files = set()
   for each_content in imdir_contents:
      if os.path.isfile(os.path.join(top_images_dir + imdir, each_content)):
         if ".png" not in each_content:
            continue
         # file looked after has number.png format
         filenum = each_content.replace(".png", "")
         if filenum.isnumeric():
            im_files.add( int(filenum) )
   
   return ( min(im_files), max(im_files) )





# shapes_dir example: C:\Users\Taichi\Documents\computer_vision\shapes\videos\street3\resized\min1\shapes\
# target_files: list of image file numbers to get image shapes for. [ 25, 26, 29 ] these image shapes will be returned
def get_all_image_shapes( shapes_dir, return_highest=False, convert_to_xy=False, target_files=None ):
   # [ lowest number image shapes, second lowest number image shapes, ... ]
   im_shapes = []
   
   
   
   target_im_dir = ""
   if convert_to_xy is True:
      # I need to get image size
      im_size = None
      
      def split_by_each_dir( param_path ):

         path_wk = param_path.split("/")
         split_path = []

         for each_dir in path_wk:
            backslash_splits = each_dir.split("\\")

            if len( backslash_splits ) >= 2:
               for backslash_split in backslash_splits:
                  split_path.append( backslash_split )
   
            else:
               split_path.append( each_dir )

         return split_path


      split_shapes_dirs = split_by_each_dir( shapes_dir )
      split_top_shapes_dir = split_by_each_dir( top_shapes_dir )

      common_dirs = set(split_shapes_dirs).intersection( set(split_top_shapes_dir) )
      split_shapes_dirs = [ temp_dir for temp_dir in split_shapes_dirs if temp_dir not in common_dirs ]

      found = True
      search_im_dir = top_images_dir
      while found is True:
         found = False
         for im_dir in os.listdir(search_im_dir):
            if im_dir == split_shapes_dirs[0]:
               split_shapes_dirs.pop(0)
               search_im_dir += im_dir + "/"
         
               found = True
        
         if found is False or len(split_shapes_dirs) == 0:
            target_im_dir = search_im_dir
            break
   
   shapes_files = os.listdir(shapes_dir)
   all_image_files = [ temp_fname for temp_fname in shapes_files if not os.path.isdir(shapes_dir + temp_fname) and "shapes.data" in temp_fname ]
   
   all_image_nums = [ int( temp_file.replace("shapes.data", "") ) for temp_file in all_image_files if temp_file.replace("shapes.data", "").isnumeric()  ]
   
   lowest_filenum = min(all_image_nums)
   highest_filenum = max(all_image_nums)
   
   ordered_shapes_files = []
   for cur_filenum in range( lowest_filenum, highest_filenum + 1 ):
      ordered_shapes_files.append( str(cur_filenum) + "shapes.data" )
   
   for cur_shape_file in ordered_shapes_files:
         
      if convert_to_xy is True and im_size is None:
         im1 = Image.open(target_im_dir + str(lowest_filenum) + ".png" )
         im_size = im1.size
      
      cur_fpath = shapes_dir + cur_shape_file
      with open (cur_fpath, 'rb') as fp:
         cur_im_shapes = pickle.load(fp)
      fp.close()
      
      if convert_to_xy is True:
         for shapeid in cur_im_shapes:
            pixels = set()
            for pindex in cur_im_shapes[shapeid]:
               xy = pixel_functions.convert_pindex_to_xy( pindex, im_size[0] )
               
               pixels.add(xy)
            
            cur_im_shapes[shapeid] = pixels
            
      im_shapes.append( cur_im_shapes )

   if return_highest is False:
      return im_shapes, lowest_filenum
   else:
      return im_shapes, lowest_filenum, highest_filenum


# im_dir example: videos\street3\resized\min1\
# target: "neighbors" -> gets image shapes neighbors data. "colors" -> gets image shapes colors 
#         "shapeids_by_pindex" -> gets data from shapes/shapeids_by_pindex/ folder.
def get_all_im_target_data( im_dir, target, min_colors=None ):

   shapes_dir = top_shapes_dir + im_dir + "shapes/"
   min_filenum, highest_filenum = get_low_highest_filenums( im_dir )
   
   all_im_target_data = []
   for cur_im_num in range(min_filenum, highest_filenum + 1 ):
      
      if target == "neighbors":
         target_file = shapes_dir + "shape_nbrs/" + str(cur_im_num) + "_shape_nbrs.data"      
      elif target == "colors":
         target_data = pixel_shapes_functions.get_all_shapes_colors(str(cur_im_num), im_dir, min_colors=min_colors)
      
      elif target == "shapeids_by_pindex":
         target_file = shapes_dir + target + "/" + str(cur_im_num) + ".data"    
      
      if target != "colors":
         with open (target_file, 'rb') as fp:
            target_data = pickle.load(fp)
         fp.close()   
      
      all_im_target_data.append( target_data )

   return all_im_target_data



# im_dir example: C:\Users\Taichi\Documents\computer_vision\images\videos\street3\resized\min1\
def get_all_image_data( im_dir ):
   # [ lowest number image data, second lowest number image data, ... ]
   all_im_data = []
   
   im_dir_contents = os.listdir(im_dir)
   # make sure that files in order. ['25.png', '26.png', ..., 'test.png', ... ]
   im_dir_contents.sort()
   
   for cur_file in im_dir_contents:
      if "png" not in cur_file or ( not cur_file.replace(".png", "").isnumeric() ):
         # file has to be number.png
         continue
      
      im_obj = Image.open( im_dir + cur_file )
      all_im_data.append( im_obj.getdata() )

   return all_im_data


# im_dir example: videos/street3/resized/min1/
def get_all_im_xy_data( im_dir ):
    # [ lowest number image data, second lowest number image data, ... ]
    all_im_data = []
   
    im_dir_contents = os.listdir(top_images_dir + im_dir)
    # make sure that files in order. ['25.png', '26.png', ..., 'test.png', ... ]
    im_dir_contents.sort()

    fnums = []
    for cur_file in im_dir_contents:
        if "png" not in cur_file or ( not cur_file.replace(".png", "").isnumeric() ):
            # file has to be number.png
            continue
        
        fnum = cur_file.replace(".png", '')
        fnums.append(int(fnum) )
    
    fnums.sort()

    for fnum in fnums:
        
        im_rgb_xy_data_fpath = top_shapes_dir + im_dir + "shapes/im_rgb_xy_data/" + str(fnum) + ".data"
        with open (im_rgb_xy_data_fpath, 'rb') as fp:
            im_rgb_xy_data = pickle.load(fp)
        fp.close()
        
        all_im_data.append(im_rgb_xy_data)
        
    return all_im_data






# total numbers to be counted down. this is the result of len( iterator such as list, set, or dictionary )
# cur_num is current number
def show_progress( total, cur_num, prev_num, remaining_chars="", in_step=1 ):

   max_len = len( str(total) )
   prev_len = len( str(prev_num) )
   
   cur_len = len(str(cur_num))
   if cur_len < prev_len:
      remaining_chars = ""
      remaining_len = max_len - cur_len
      for char_len in range( remaining_len ):   
         remaining_chars += " "
   

   if cur_num % in_step == 0:
      print("\r",cur_num, remaining_chars, " remaining", end="")

   if cur_num == in_step:
      # last 
      print( )

   return remaining_chars


# when separate_1tomulti_matches.py updated im_shapes below is the consequence
# image pixels no longer belong to just only one shape but some of them may belong to multiple shapes.
# this is still true that all image pixels belong one or more shapes and there is no image pixel that does that belong to any shape.
#
# operation specifies crud. create, read, update, delete.
# for update, delete operations. make sure to update im_shapes before calling this function.
def create_shapeids_by_pindex( im_fname, directory, im_size ):

   shapes_dir = top_shapes_dir + directory + "shapes/"
   shapes_dfile = shapes_dir + im_fname + "shapes.data"
   with open (shapes_dfile, 'rb') as fp:
      # { '79999': ['79999', ... ], ... }
      # { 'shapeid': [ pixel indexes ], ... }
      im_shapes = pickle.load(fp)
   fp.close()

   shp_by_index_dir = shapes_dir + 'shapeids_by_pixel/'
   if not os.path.isdir(shp_by_index_dir):
      os.makedirs(shp_by_index_dir)

   shp_by_index_dfile = shp_by_index_dir + im_fname + ".data"

   image_pixels_len = im_size[0] * im_size[1]
   
   shapeids_by_pindex = {}
   for pindex in range(image_pixels_len):
      shapeids_by_pindex[pindex] = set()
   
   for shapeid in im_shapes:
      for pindex in im_shapes[shapeid]:
         shapeids_by_pindex[pindex].add(shapeid)


   with open(shp_by_index_dfile, 'wb') as fp:
      pickle.dump(shapeids_by_pindex, fp)
   fp.close()  






























