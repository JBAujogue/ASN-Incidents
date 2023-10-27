import os
import unicodedata
import re


import docx
from docx import Document
from docx.text.paragraph import Paragraph
from docx.text.run import Run
from docx.table import Table

from docx.oxml.ns import qn
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.enum.style import WD_STYLE_TYPE




# -------------------------- Constants ----------------------------- 
EXCLUDE_STYLES = ['Title',
                  'toc 1',
                  'toc 2',
                  'toc 3',
                  'toc 4',
                  'toc 5',
                  'toc 6',
                  'toc 7',
                  'toc 8',
                  'toc 9',
                  'toc 10',
                  'wcp_TOCTitle',
                  'wcp_Lists',
                  'wcp_HiddenText',
                  'wcp_SubTitle',
                  'wcp_Abbreviation',
                  'table of figures',
                  'Subtitle',
                  'wcp_BibliographicReference']

TOC_STYLES = ['toc 1',
              'toc 2',
              'toc 3',
              'toc 4',
              'toc 5',
              'toc 6',
              'toc 7',
              'toc 8',
              'toc 9',
              'toc 10',
              'wcp_TOCTitle',
              'table of figures']

TABLE_STYLES = ['wcp_AttachmentTitle', 'wcp_CaptionTable', 'wcp_TableRowHeader']

ASTERISK_STYLES = ['wcp_TableContentSmall', 'wcp_TableContent', 'wcp_Tablenote']

BULLET_STYLES = ['List Number', 
                 'List Number 2', 
                 'List Number 3', 
                 'List Number 4', 
                 'List Bullet', 
                 'List Bullet 2', 
                 'List Bullet 3', 
                 'List Bullet 4',
                 'ListSubText',
                 'ListSubText2',
                 'ListSubText3',
                 'ListSubText4'
                 'wcp_ListSubText',
                 'wcp_ListSubText2',
                 'wcp_ListSubText3',
                 'wcp_ListSubText4']

EXCLUDE_TEXTS =['List of Tables',
                'List of Table',
                'Liste of Tables',
                'Liste of Table',
                'List of Figures',
                'List of Figure',
                'Liste of Figures',
                'Liste of Figure',
                'Table of Contents', 
                'Table of Content']




# ------------------------------ Runs ------------------------------ 
def keep_run(run):
    return (
        'wcp_HiddenText' not in run.style.name and 
        'wcp_WisdomInternal' not in run.style.name and (
            'wcpc_AuthoringInstruction' not in run.style.name 
            or [vt for vt in run._element.xpath('.//w:vanish') if vt.val == False] != []
        )
        and (
            run._element.xpath('.//w:vanish') == [] 
            or [vt for vt in run._element.xpath('.//w:vanish') if vt.val == False] != []
        ) 
        and get_run_text(run) != ''
    )



def get_run_text(run, normalize = True) :
    """
	taken from 
    https://github.com/BayooG/bayoo-docx/blob/8df4039d29fe12fc7d8f6731780f4575f99f2f6e/docx/oxml/text/run.py

    A string representing the textual content of this run, with content
    child elements like ``<w:tab/>`` translated to their Python
    equivalent.
    """
    text = ''
    for child in run._r:
        if child.tag == qn('w:t'):
            t_text = child.text
            text += (t_text if t_text is not None else '')
        elif child.tag == qn('w:tab'):
            text += '\t'
        elif child.tag in (qn('w:br'), qn('w:cr')):
            text += '\n'
        elif child.tag == qn('w:noBreakHyphen'):
            text += '-'
    if normalize : text = unicodedata.normalize("NFKD", text)
    return text




# ---------------------------- Blocks ------------------------------ 
def make_element(el, doc):
    if   isinstance(el, CT_P)  : return Paragraph(el, doc)
    elif isinstance(el, CT_Tbl): return Table(el, doc)
    return



def get_blocks(doc):
    blocks = [make_element(el, doc) for el in doc.element.body.iterchildren()] #doc._element[0]]
    blocks = [b for b in blocks if b is not None]
    return blocks



def keep_block(block) :
     return (
        (block.text.lower() not in [ex.lower() for ex in EXCLUDE_TEXTS]) 
        and (block.style.name not in EXCLUDE_STYLES)
     )



def get_runs(block):
    '''variant of python-docx attribute block.runs, 
       as this attributes delivers additional hidden text,
       and may miss other text.
    '''
    runs = [Run(r, block) for r in block._element.xpath('.//w:r')]
    return [r for r in runs if keep_run(r)]



def get_block_text(block, normalize = True) :
    runs = get_runs(block)
    block_text = ''.join([get_run_text(run, normalize) for run in runs]).strip()
    return block_text





# ---------------------------- Tables ------------------------------ 
def get_table_content(table):
    '''returns the content of a table as a 2D [[str]] with :

        - dim 1 corresponding to row number of the cell
        - dim 2 corresponding to column number of the cell
        - str corresponding to the cell text
    '''
    content = [
        [[] for j, col in enumerate(table.columns)] 
        for i, row in enumerate(table.rows)
    ]
    for cell in table._cells: 
        content[cell._tc.top][cell._tc.right -1] = get_cell_text(cell)
    return content



def get_cell_blocks(cell, paragraph_only = False) :
    if paragraph_only : blocks = [make_element(el, cell) for el in cell._tc.iterchildren() if isinstance(el, CT_P)]
    else              : blocks = [make_element(el, cell) for el in cell._tc.iterchildren()]
    blocks = [b for b in blocks if b is not None]
    return blocks



def get_cell_text(cell) :
    blocks = get_cell_blocks(cell, paragraph_only = True)
    text = ' '.join([get_block_text(b) for b in blocks])
    return text





# ------------------------------ TOC ------------------------------- 
def is_there_a_toc(doc) :
    full_content = [Paragraph(el, doc) for el in doc._element[0] if isinstance(el, CT_P)]
    for block in full_content :
        if block.style.name in TOC_STYLES: 
            return True
    return False



def get_toc_by_depth(doc, symbol = ' --- '):
    '''extract the Table Of Content from a docx file, under the form of a list

       [[0, title_of_document],
        [1, section_title],
        [2, subsection_title],
        [2, subsection_title],
        [3, paragraph_title],
        ...]

       Each pair corresponds to a section / subsection / paragraph title, 
       the digit corresponds to the depth of the title (heading 1, heading 2, heading 3 etc) in the docx file, 
       and the string gives the content of the title
    '''
    toc = []
    full_title = ''
    full_title_found = False
    last_num = 0
    blocks = get_blocks(doc)
    toc_present = is_there_a_toc(doc)

    for block in blocks:
        if block.__class__.__name__ == 'Paragraph':
            block_text = get_block_text(block)

            # if the document title is not completely parsed
            if not full_title_found:

                # if the block is a part of the document title, append it
                if block.style.name in ['Title', 'Subtitle', 'wcp_SubTitle'] and block_text != '' :
                    full_title += ((symbol if full_title != '' else '') + block_text)

                # if text is centered and the block is not a TOC, append it to the full title
                elif (
                    (block.alignment == docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER or toc_present)
                    and (block.style.name not in TOC_STYLES)
                    and sum([ex in block_text for ex in EXCLUDE_TEXTS]) == 0
                    and keep_block(block)
                    and block_text != ''
                    ):
                    if 'Subtitle' in [sty.name for sty in doc.styles if sty.name] : 
                        block.style = doc.styles['Subtitle']
                    
                    elif 'wcp_Subtitle' in [sty.name for sty in doc.styles if sty.name] : 
                        block.style = doc.styles['wcp_Subtitle']
                    
                    else :
                        doc.styles.add_style('Subtitle', WD_STYLE_TYPE.PARAGRAPH)
                        block.style = doc.styles['Subtitle']
                
                    full_title += ((symbol if full_title != '' else '') + block_text)

                # if the block is a new section title or TOC,
                # cumulate the document title
                elif (block.style.name[:7] == 'Heading') \
                  or (block.style.name in TOC_STYLES) \
                  or sum([ex in block_text for ex in EXCLUDE_TEXTS]) > 0 :
                    toc.append([0, full_title])
                    full_title_found = True

            # if the block is a section/subsection of the corp
            if ((block.style.name[:7] == 'Heading' or 'SubHeading' in block.style.name) 
                and block_text != ''
                ): 
                if block.style.name[:7] == 'Heading' : 
                    last_num = int(block.style.name[8])
                    toc.append([last_num, block_text])

                elif 'SubHeading' in block.style.name: 
                    toc.append([last_num + 1, block_text])
                
    if toc == []: 
        toc.append([0, full_title])
    return toc



def get_toc(doc, symbol = ' --- '):
    '''extract the Table Of Content from a docx file, under the form of a list

       [[0, title_of_document],
        [1, section_title],
        [1.1, subsection_title],
        [1.2, subsection_title],
        [1.2.1, paragraph_title],
        ...]

       Each pair corresponds to a section / subsection / paragraph title, 
       the digit corresponds to the numerotation of the title in the docx file, 
       and the string gives the content of the title
    '''
    toc = []
    toc_by_depth = get_toc_by_depth(doc, symbol)
    counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for title_depth, title in toc_by_depth:
        # get title counter
        counter[title_depth] += 1
        for j in range(title_depth + 1, len(counter)): 
            counter[j] = 0
        
        # convert counter to string
        title_num = '.'.join([str(c) for c in counter[1:] if c>0])
        toc.append([title_num, title])
    return toc



def make_toc_as_dict(toc):
    '''extract the Table Of Content from a docx file, under the form of a dict

       {'0' : [title_of_document],
        '1' : [title_of_document, section_title],
        '1.1' : [title_of_document, section_title, subsection_title],
        '1.2' : [title_of_document, section_title, subsection_title],
        '1.2.1' : [title_of_document, section_title, subsection_title, paragraph_title],
        ...}

       Each pair corresponds to a section / subsection / paragraph title, 
       the digit corresponds to the numeration of the title in the docx file, 
       and the list gives the content of each parent title
    '''
    atoc = {title_num: [
            title2 for title_num2, title2 in toc
            if title_num.startswith(title_num2)
        ] 
        for title_num, _ in toc
    }
    return atoc



def get_title(toc, atoc, n_title) :
    return [toc[n_title][0], ' | '.join(atoc[toc[n_title][0]])]






# --------------------------- Document ----------------------------- 
def parse_docx(doc, symbol = ' --- '):
    '''Returns the content of 'doc' as a list of paragraphs, each paragraph displayed as

                [paragraph number, paragraph full_title, content] 

       where : 
           - the paragraph full title is the concatenation of section, paragraph and sub-paragraph titles with '|' as separator
           - the content is the list of sentences contained in the paragraph
           - each sentence has the form [sentence_object, sentence_text] where sentence_objet is a run in the python-docx vocabulary
    '''
    toc    = get_toc(doc, symbol)
    atoc   = make_toc_as_dict(toc)
    blocks = get_blocks(doc)

    processed_content = []
    tables = {}
    n_title = 0
    key = ''
    current_content = []

    # for each block
    for block in blocks : 
        
        # if the block is a non-empty Paragraph to keep
        if (block.__class__.__name__ == 'Paragraph' 
            and keep_block(block) 
            and get_block_text(block) != ''
            ): 

            # if the block is a title
            if block.style.name[:7] == 'Heading' or 'SubHeading' in block.style.name :
                
                # append previous title and text into processed content
                title = get_title(toc, atoc, n_title) # [section_num, section_full_title]
                processed_content.append(title + [current_content])
                
                # increment to current title
                n_title += 1
                current_content = []

            # if the block is not a title
            else :
                runs = get_runs(block)
                text = get_block_text(block, normalize = True)

                # if the block is an asterisk
                if (block.style.name in ASTERISK_STYLES and key != '') :
                    adds = key + ' | '
                else :
                    adds = ''

                block_content = [[run, adds + get_run_text(run).strip()] for run in runs]
                current_content.append([block, block_content])

                # if the block is a table name
                bool1 = block.style.name[:6] == 'Captio' and 'Table' in text # or 'Figure' in text)
                bool2 = block.style.name[:6] == 'Normal' and re.match('Table[s\s]*[0-9]+[\s]*:', text)
                bool3 = block.style.name in TABLE_STYLES and re.match('Table[s\s]*[0-9]+[\s]*:', text)
                if bool1 or bool2 or bool3: 
                    key = text
                    tables[key] = []

        # if the block is a table and some table title was previously found, 
        # cumulate table content
        elif (block.__class__.__name__ == 'Table' and key != ''):
            tables[key].append([block, get_table_content(block)])

    # append last paragraph
    title = get_title(toc, atoc, n_title)
    processed_content.append(title + [current_content])

    # remove empty tables
    tables = {t_name: t for t_name, t in tables.items() if t != []}

    output = {
        'toc' : toc, 
        'atoc' : atoc, 
        'content' : processed_content, 
        'tables' : tables, 
        'styles' : doc.styles,
    }
    return output