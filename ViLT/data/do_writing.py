from write_probing_data import make_arrow
from write_probing_flowers import make_arrow_flowers
make_arrow('bshift', 'probing', "flickr")
make_arrow('bshift_dec', 'probing', "flickr")
make_arrow('size', 'probing', "flickr")
make_arrow('size_dec', 'probing', "flickr")
make_arrow('pos', 'probing', "flickr")
make_arrow('pos_dec', 'probing', "flickr")
make_arrow('colors', 'probing', "flickr")
make_arrow('colors_dec', 'probing', "flickr")
make_arrow('objcount', 'probing', "coco")
make_arrow('objcount_dec', 'probing', "coco")
make_arrow('altcap', 'probing', "coco")
make_arrow('altcap_dec', 'probing', "coco")
make_arrow_flowers('probing')
make_arrow('tag', 'probing', "flickr")
make_arrow('tag_dec', 'probing', "flickr")

