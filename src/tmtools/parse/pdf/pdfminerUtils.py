
import sys
import os

# for data
import pandas as pd
import numpy as np

# for pdf parsing
import pdfminer
from pdfminer.high_level import extract_text, extract_pages, extract_text_to_fp
from pdfminer.layout import LAParams




#**********************************************************
#*                      functions                         *
#**********************************************************

# ------------------------- Utils -------------------------
# bbox format:
# bbox = (0, 0, 0, 0) is the LOWER left corner
# bbox = (x_left, y_bottom, x_right, y_top)

def reorder_text_pdfminer(line):
    def get_x0(line, i):
        try:
            l_x0 = line[i-1].x0
        except:
            l_x0 = None
        try:
            r_x0 = line[i+1].x0
        except:
            r_x0 = None
        if l_x0 and r_x0:
            x0 = (l_x0 + r_x0)/2
        elif l_x0:
            x0 = l_x0
        elif r_x0:
            x0 = r_x0
        return x0
    
    # strip line
    line = list(line)
    inds = [i for i in range(len(line)) if isinstance(line[i], pdfminer.layout.LTChar)]
    line = [line[i] for i in range(min(inds), max(inds)+1)]
    
    # get characters
    chars = [c.get_text() for c in line]
    
    # get each character x0 coordinate
    inds = [
        (line[i].x0 if isinstance(line[i], pdfminer.layout.LTChar) else get_x0(line, i)) 
        for i in range(len(line))
    ]
    # sort characters by x0
    line = [(i, c) for i, c in zip(inds, chars)]
    line.sort(key = lambda ic: ic[0])
    text = ''.join([ic[1] for ic in line])
    return text


def join_lines(df):
    def shrink_group(group):
        page_num = int(group[0][0])
        rect_y = group[0][2]
        d_srt = {line[1]: line[3] for line in group}
        d_end = dict(d_srt)
        ended = False
        while not ended:
            d_end = {k: (d_srt[v] if v in d_srt else v) for k, v in d_srt.items()}
            ended = (d_end == d_srt)
            d_srt = dict(d_end)
        
        vals = list(set(d_end.values()))
        d_inv = {v: min([k for k in d_end if d_end[k] == v]) for v in vals}
        group = [[page_num, v, rect_y, k, rect_y] for k, v in d_inv.items()]
        return group
        
    # select lines, e.g. rectangles with y0 == y1 
    df_rect = df[list(df.rect_y0 != df.rect_y1)]
    df_line = df[list(df.rect_y0 == df.rect_y1)]
    # df_rect = df[list(abs(df.rect_y0 - df.rect_y1) > approx)]
    # df_line = df[list(abs(df.rect_y0 - df.rect_y1) <= approx)]
    
    if df_line.shape[0]>0:
        # group lines that are aligned, e.g. with same page_num and y coordinate
        groups = df_line.groupby(['page_num', 'rect_y0'])

        # fusion contiguous lines
        joined_lines = groups.apply(lambda g: shrink_group(g.values.tolist()))
        joined_lines = [lines for group in joined_lines.values.tolist() for lines in group]

        # merge result
        df_line = pd.DataFrame(joined_lines, columns = df.columns)
        df = pd.concat((df_rect, df_line), ignore_index = True)\
            .sort_values(['page_num', 'rect_y1', 'rect_x0'])\
            .reset_index(drop = True)
    else:
        df = df_rect
    return df



# ------------------------ Parsing ------------------------
def parse_pdf_pdfminersix(doc, precision = 4, swap_yaxis = True):
    parsed_text = []
    parsed_rect = []
    parsed_imgs = []
    doc_name = doc.split(os.path.sep)[-1]
    pages = extract_pages(doc)
    for page_idx, page in enumerate(pages):
        page_x, page_y = page.bbox[-2:]
        rect = [page_x, page_y, page_x, page_y]
        # swap values along y0 and y1 to get top-down y-axis numbering
        swap = ([0, 1, 0, 1] if swap_yaxis else [0, 0, 0, 0])
        for block_idx, block in enumerate(page):
            # a block can be one of the following subclass of pdfminer.layout :
            # LTTextBox, LTFigure, LTLine, LTRect or an LTImage
            # other classes exist, see help(pdfminer.layout)
            
            # get bbox, with :
            # - normalized x and y coordinates (eg float values ranging in [0, 1])
            #block_x0, block_y1, block_x1, block_y0 = block.bbox
            block_x0, block_y1, block_x1, block_y0 = [
                round(abs(s-v/w), precision) for s, v, w in zip(swap, block.bbox, rect)
            ]
            # parse text
            if type(block) in [
                pdfminer.layout.LTTextBoxHorizontal, 
                pdfminer.layout.LTTextBoxVertical
                ]:
                for line_idx, line in enumerate(block):
                    line_x0, line_y1, line_x1, line_y0 = [round(abs(s-v/w), precision) for s, v, w in zip(swap, line.bbox, rect)]
                    text = line.get_text().strip()
                    if text != '':
                        text_reordered = reorder_text_pdfminer(line).strip()
                        parsed_text.append([
                            doc_name, page_idx, block_idx, line_idx, 
                            text, text_reordered,
                            block_x0, block_y0, block_x1, block_y1,
                            line_x0, line_y0, line_x1, line_y1,
                        ])
            
            # parse images
            elif type(block) == pdfminer.layout.LTImage:
                parsed_imgs.append([
                    page_idx, block_idx,
                    block_x0, block_y0, block_x1, block_y1,
                ])
            
            # figures are bags of characters, hence difficult to treat
            elif type(block) == pdfminer.layout.LTFigure:
                continue
            
            # parse rect and lines (rects with 0 width)
            elif type(block) in [
                pdfminer.layout.LTLine,
                pdfminer.layout.LTRect,
                ]:
                parsed_rect.append([page_idx, block_x0, block_y0, block_x1, block_y1])


    df_text = pd.DataFrame(parsed_text, columns = [
        'doc_name', 'page_num', 'block_num', 'line_num', 
        'text', 'text_reordered', 
        'block_x0', 'block_y0', 'block_x1', 'block_y1',
        'line_x0', 'line_y0', 'line_x1', 'line_y1',
    ])
    df_text['block_y0'] += df_text.page_num
    df_text['block_y1'] += df_text.page_num
    df_text['line_y0']  += df_text.page_num
    df_text['line_y1']  += df_text.page_num
    df_text['center_x'] = (df_text.line_x1 + df_text.line_x0)/2
    df_text['center_y'] = (df_text.line_y1 + df_text.line_y0)/2

    df_imgs = pd.DataFrame(parsed_imgs, columns = [
        'page_num', 'block_num', 
        'img_x0', 'img_y0', 'img_x1', 'img_y1',
    ])
    df_imgs['img_y0'] += df_imgs.page_num
    df_imgs['img_y1'] += df_imgs.page_num
    df_rect = pd.DataFrame(parsed_rect, columns = [
        'page_num', 'rect_x0', 'rect_y0', 'rect_x1', 'rect_y1',
    ])
    df_rect['rect_y0'] += df_rect.page_num
    df_rect['rect_y1'] += df_rect.page_num
    df_rect = join_lines(df_rect)
    return (df_text, df_imgs, df_rect)

