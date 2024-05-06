OPRID = {'ACD':	'Accordion',
         'AB': 'Air Bubbles',
         'AO': 'Anneal Off',
         'ASL': 'Anneal Segment Length',
         'BC': 'Bad Cut',
         'ВP': 'Lump/Bump',
         'BW': 'Broken Wire',
         'CRK': 'Cracks',
         'DF': 'Overmelt',
         'DH': 'Damaged Hypotube',
         'DIS':	'Damaged Inner Surface',
         'DL': 'Damaged Liner',
         'DS': 'Damaged Shaft',
         'DSU': 'Damaged Surface',
         'DT': 'Damaged Tip',
         'DW': 'Damaged Wire',
         'DEL':	'Delamination',
         '#1US': 'Dim 1 Outer Diameter Oversized',
         '#1OS':	'Dim 1 Outer Diameter Undersized',
         '#2OS':	'Dim 2 Inner Diameter Oversized',
         '#2US': 'Dim 2 Inner Diameter Undersized',
         '#5OS': 'Dim 5 Length Oversized',
         '#5US':	'Dim 5 Length Undersized',
         '#6OS':	'Dim 6 Outer Diameter Oversized',
         '#6US': 'Dim 6 Outer Diameter Undersized',
         '#7OS':	'Dim 7 Length Oversized',
         '#7US':	'Dim 7 Length Undersized',
         '#9OS': 'Dim 9 Outer Diameter Oversized',
         '#9US':	'Dim 9 Outer Diameter Undersized',
         'DISC':	'Discoloration',
         'DST': 'Distortion',
         'DU': 'Dropped Unit',
         'EXG':	'Exit Gate Off',
         'EH': 'Exposed Hypotube',
         'EW': 'Exposed Wire',
         'LT': 'Failed Leak Test',
         'FL': 'Flash',
         'FM': 'Foreign Material',
         'FB': 'Fibers',
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
         'SP': 'Samples',
         'SCR':	'Scratches',
         'SLOS': 'Segment Length Long',
         'SLUS': 'Segment Length Short',
         'SW': 'Short Wire',
         'SKV': 'Skive Mark',
         'SKVD': 'Skive Damage',
         'STN': 'Stain',
         'STR': 'Stringers',
         'TT': 'Tensile Test',
         'TL': 'Tails',
         'TL': 'Tight Liner',
         'TD': 'Tip Damaged',
         'VD': 'Voids',
         'WL': 'White Line',
         'WS':	'White Spot Over Stretch',
         'WC': 'Window Close (SL/US)',
         'WM':	'Window Miscut',
         'OW':	'Window Open (SL/OS)',
         'WB':	'Wire Bent',
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
