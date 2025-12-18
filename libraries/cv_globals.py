import os

debug = None

os.chdir("./")

proj_dir = os.getcwd()

if proj_dir != "" and proj_dir[-1] != "/":
   proj_dir +='/'

top_shapes_dir = proj_dir + "shapes/"
top_images_dir = proj_dir + "images/"
top_tools_dir = proj_dir + "tools/"
top_tests_dir = proj_dir + "tests/"
top_temp_dir = proj_dir + "temp/"

test_im_fpath = ""
def store_test_im_fpath( param_test_im_fpath ):
	global test_im_fpath
	test_im_fpath = param_test_im_fpath

def get_test_im_fpath():
	return test_im_fpath

def store_debug(param_debug):
	global debug
	debug = param_debug

def get_debug():
	return debug



# default match threshold
m_threshold = 0.6


def get_smallest_pixc( im_size ):
   smallest_pixc = ( im_size[0] * im_size[1] ) / 20000
   return smallest_pixc

def get_sec_smallest_pixc( im_size ):
   sec_smallest_pixc = ( im_size[0] * im_size[1] ) / 11428
   return sec_smallest_pixc

def get_third_smallest_pixc( im_size ):
   third_smallest_pixc = ( im_size[0] * im_size[1] ) / 6154
   return third_smallest_pixc

def get_frth_smallest_pixc( im_size ):
   frth_smallest_pixc = ( im_size[0] * im_size[1] ) / 3809
   return frth_smallest_pixc

def get_fifth_s_pixc( im_size ):
   fifth_s_pixc = ( im_size[0] * im_size[1] ) / 2857
   return fifth_s_pixc

def get_6th_s_pixc( im_size ):
   sixth_s_pixc = ( im_size[0] * im_size[1] ) / 2424
   return sixth_s_pixc

def get_svnth_pixc( im_size ):
   svnth_s_pixc = ( im_size[0] * im_size[1] ) / 1951
   return svnth_s_pixc

def get_Lshape_size( im_size ):
   Lshape_size = ( im_size[0] * im_size[1] ) / 1600
   return Lshape_size

def get_indpnt_shape_size( im_size ):
   # independent shape size
   indpnt_shape_size = ( im_size[0] * im_size[1] ) / 1230
   return indpnt_shape_size

def get_extraL_shp_size(im_size):
	extra_large_size = ( im_size[0] * im_size[1] ) / 950
	return extra_large_size



def get_default_color_threshold():
   return 30

def get_smallest_mv_threshold( im_size ):
   smallest_mv_threshold = ( im_size[0] * im_size[1] ) / 40000
   return smallest_mv_threshold














