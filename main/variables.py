OPRID = {'ACD':	'Accordion',
         'AB': 'Air Bubble',
         'AB': 'Air Bubbles',
         'AO': 'Anneal Off',
         'ASL': 'Anneal Segment Length',
         'BC': 'Bad Cut',
         'BW': 'Broken Wire',
         'CRK': 'Cracks',
         'DH': 'Damaged Hypotube',
         'DIS':	'Damaged Inner Surface',
         'DL': 'Damaged Liner',
         'DS': 'Damaged Shaft',
         'DSU': 'Damaged Surface',
         'DT': 'Damaged Tip',
         'DW': 'Damaged Wire',
         'DEL':	'Delamination',
         '#01US': 'Dim 1 Outer Diameter Oversized',
         '#01OS':	'Dim 1 Outer Diameter Undersized',
         '#02OS':	'Dim 2 Inner Diameter Oversized',
         '#02US': 'Dim 2 Inner Diameter Undersized',
         '#05OS': 'Dim 5 Length Oversized',
         '#05US':	'Dim 5 Length Undersized',
         '#06OS':	'Dim 6 Outer Diameter Oversized',
         '#06US': 'Dim 6 Outer Diameter Undersized',
         '#07OS':	'Dim 7 Length Oversized',
         '#07US':	'Dim 7 Length Undersized',
         '#09OS': 'Dim 9 Outer Diameter Oversized',
         '#09US':	'Dim 9 Outer Diameter Undersized',
         'DISC':	'Discoloration',
         'DST': 'Distortion',
         'DU': 'Dropped Unit',
         'EXG':	'Exit Gate Off',
         'EH': 'Exposed Hypotube',
         'EW': 'Exposed Wire',
         'LT': 'Failed Leak Test',
         'FL': 'Flash',
         'FM': 'Foreign Material',
         'GAP':	'Gap',
         'GD': 'Glue Damage',
         'HOS': 'Heat Off Stopper',
         'TEMP':	'Incorrect Tempeature',
         'IDB': 'Inner Diameter Block',
         'IDBO':	'Inner Diameter Blow Out',
         'LT': 'Leak Test',
         'LS': 'Length Short',
         'LPM': 'Loose Particulate Matter',
         'MOF':	'Material Overflow',
         'MP':	'Micropores',
         'MA': 'Misaligned',
         'MAS': 'Misassemble',
         'MAS':	'Misassembled',
         'MEX': 'Misassembled Extrusions',
         'OF': 'Over Flow',
         'OA': 'Overalign',
         'OAL OS': 'Overall Length Long',
         'OAL US':	'Overall Length Short',
         'OF': 'Overflow',
         'PC': 'Pancake',
         'PROT': 'Protrusion',
         'GN11': 'PTFE on Pull Wire',
         'RM': 'Raised Material',
         'RG': 'Reglue',
         'RG': 'Re-glue',
         'SP': 'Samples',
         'SCR': 'Scratch',
         'SCR':	'Scratches',
         'SL OS': 'Segment Length Long',
         'SL US': 'Segment Length Short',
         'SW': 'Short Wire',
         'SKV': 'Skive Mark',
         'SKVD': 'Skive Damage',
         'STN': 'Stain',
         'STR': 'Stringers',
         'TL': 'Tails',
         'TT': 'Tensile Test',
         'TL': 'Tight Liner',
         'TD': 'Tip Damage',
         'TD': 'Tip Damaged',
         'VD': 'Voids',
         'WL': 'White Line',
         'WS':	'White Spot Over Stretch',
         'WC': 'Window Close (SL/US)',
         'WM':	'Window Miscut',
         'OW':	'Window Open (SL/OS)',
         'WB':	'Wire Bent',
         'WB':	'Wire Bent ',
         'WO':	'Wire Out',
         'WK': 'Wrinkles'}
COLUMNHEADING = ['Work Center',
                 'Operation',
                 'Scrap Code',
                 'Scrap Description',
                 'Op. Good Qty',
                 'Op. Scrap Qty',
                 'UoM',
                 'PPM____________',
                 'Posting date',
                 'Entry Date',
                 'Prod Order',
                 'Material Number',
                 'Material Description',
                 'Parent Good qty',
                 'Parent Scrap qty',
                 'Order Unit',
                 'Order Type',
                 'Plant',
                 'Entered Good Qty',
                 'Entered Scrap Qty',
                 'Entered UoM']
'''
Work Center: column 2
Operation: column 1
Scrap Code: correspond with Scrap Description
Scrap Description: column 4, need to expand the short form 
Op. Good Qty: column 3
Op. Scrap Qty: minus by Op. Good Qty
'''
