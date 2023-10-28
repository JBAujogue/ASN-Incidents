
import sys
import os
from io import open, BytesIO, StringIO

# for data
import pandas as pd
import numpy as np

# image parsing
try:
    from PIL import Image
except ImportError:
    import Image

# for pdf parsing
import fitz # name for the PyMuPDF package

# custom package
import pytesseract
from tesseractUtils import parse_image_tesseract




#**********************************************************
#*                      functions                         *
#**********************************************************

# ------------------------- Utils -------------------------
# bbox format:
# bbox = (0, 0, 0, 0) is the UPPER left corner
# bbox = (x_left, y_top, x_right, y_bottom)

def get_reordered_text_pymupdf(span, line_dir):
    chars = list(span['chars'])
    chars.sort(key = lambda c: float(c['bbox'][0])*line_dir) #origin
    text = ''.join([c['c'] for c in chars])
    return text



# ------------------------ Parsing ------------------------
def parse_pdf_pymupdf(
        folder_name,
        file_name,
        images_folder_name = None,
        precision = 4,
    ):
    full_name = os.path.join(folder_name, file_name)
    
    # create folder for images
    if images_folder_name:
        imgs_path = os.path.join(folder_name, images_folder_name)
        if not os.path.isdir(imgs_path): 
            os.mkdir(imgs_path)
        imgs_path = os.path.join(imgs_path, file_name[:-4])
        if not os.path.isdir(imgs_path): 
            os.mkdir(imgs_path)
        if not os.path.isdir(imgs_path + ' (pdf)'): 
            os.mkdir(imgs_path + ' (pdf)')
            
    
    parsed_text = []
    parsed_imgs = []
    with fitz.open(full_name) as doc:
        # iterate over pages
        # list of methods for Page objects
        # see the docs: https://pymupdf.readthedocs.io/en/latest/page.html
        for page_idx, page in enumerate(doc):
            page_x, page_y = page.rect[-2:]
            rect = [page_x, page_y, page_x, page_y]

            # extract images
            # see the docs: https://pymupdf.readthedocs.io/en/latest/faq.html#how-to-extract-images-pdf-documents
            if images_folder_name:
                images = page.get_images(full = True)
                for image_idx, image_obj in enumerate(images):
                    image_xref = image_obj[0]
                    image_dict = doc.extract_image(image_xref)
                    image_ext  = image_dict['ext']
                    image_name = 'image {}_{}'.format(page_idx, image_idx)
                    image_byte = BytesIO(image_dict['image'])

                    with Image.open(image_byte) as image:
                        # see https://ai-facets.org/tesseract-ocr-best-practices/
                        df_img = parse_image_tesseract(
                            image, 
                            lang = 'eng+fra', 
                            output_type= 'dict',
                            config = '--oem 1 --psm 12'
                        )
                        if images_folder_name:
                            # export images
                            image_path = os.path.join(imgs_path, image_name + '.' + image_ext)
                            image.save(open(image_path, "wb"))

                            # export pdf-converted images
                            image_pdf  = pytesseract.image_to_pdf_or_hocr(image, extension = 'pdf')
                            image_path = os.path.join(imgs_path + ' (pdf)', image_name + '.pdf')
                            with open(image_path, 'w+b') as f:
                                f.write(image_pdf)
        
            # get positions of images
            img_infos = page.get_image_info()
            for info_idx, info in enumerate(img_infos):
                img_x0, img_y0, img_x1, img_y1 = [round(v/w, precision) for v, w in zip(info['bbox'], rect)]
                parsed_imgs.append([page_idx, info_idx, img_x0, img_y0, img_x1, img_y1])
            
            # get content and position of texts
            textpage = page.get_textpage()      # convert Page to TextPage object
            content = textpage.extractRAWDICT() # get page content as dict
            # equivalently :
            #content = page.get_text(('rawdict'))
            # alternatively :
            #content = textpage.extractDICT()   # get page content as dict
            # equivalently :
            #content = page.get_text(('dict'))
            
            # blocks aren't sorted by top-to-bottom order !!
            # lines aren't sorted by top-to-bottom order !!
            
            blocks = list(content['blocks'])
            blocks = [b for b in blocks if 'lines' in b]
            blocks.sort(key = lambda b: (
                min([float(span['bbox'][3]) for l in b['lines'] for span in l['spans']]),
                min([float(span['bbox'][0]) for l in b['lines'] for span in l['spans']])
            ))
            for block_idx, block in enumerate(blocks):
                block_x0, block_y0, block_x1, block_y1 = [round(v/w, precision) for v, w in zip(block['bbox'], rect)]
                lines = list(block['lines'])
                lines.sort(key = lambda l: (
                    min([float(span['bbox'][3]) for span in l['spans']]),
                    min([float(span['bbox'][0]) for span in l['spans']])
                ))
                for line_idx, line in enumerate(lines):
                    line_x0, line_y0, line_x1, line_y1 = [round(v/w, precision) for v, w in zip(line['bbox'], rect)]
                    spans = line['spans']
                    
                    for span_idx, span in enumerate(spans):
                        span_x0, span_y0, span_x1, span_y1 = [round(v/w, precision) for v, w in zip(span['bbox'], rect)]
                        text = ''.join([c['c'] for c in span['chars']])
                        text_reordered = get_reordered_text_pymupdf(span, line['dir'][0])
                        font = span['font']
                        size = span['size']

                        parsed_text.append([
                            page_idx, block_idx, line_idx, span_idx,
                            text, text_reordered, font, size,
                            block_x0, block_y0, block_x1, block_y1,
                            line_x0, line_y0, line_x1, line_y1,
                            span_x0, span_y0, span_x1, span_y1,
                        ])

    df_text = pd.DataFrame(parsed_text, columns = [
        'page_num', 'block_num', 'line_num', 'span_num',
        'text', 'text_reordered', 'font', 'fontsize',
        'block_x0', 'block_y0', 'block_x1', 'block_y1',
        'line_x0', 'line_y0', 'line_x1', 'line_y1',
        'span_x0', 'span_y0', 'span_x1', 'span_y1',
    ]) 
    df_text['block_y0'] += df_text.page_num
    df_text['block_y1'] += df_text.page_num
    df_text['line_y0']  += df_text.page_num
    df_text['line_y1']  += df_text.page_num
    df_text['span_y0']  += df_text.page_num
    df_text['span_y1']  += df_text.page_num
    df_text['center_x'] = (df_text.span_x1 + df_text.span_x0)/2
    df_text['center_y'] = (df_text.span_y1 + df_text.span_y0)/2

    df_imgs = pd.DataFrame(parsed_imgs, columns = [
        'page_num', 'img_num', 'img_x0', 'img_y0', 'img_x1', 'img_y1'
    ]) 
    df_imgs['img_y0'] += df_imgs.page_num
    df_imgs['img_y1'] += df_imgs.page_num
    return (df_text, df_imgs)

