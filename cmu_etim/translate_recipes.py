import numpy as np
import os.path
import pandas as pd
import sklearn.model_selection


def clean_recipes(path: str = 'data/epi_r.csv', save_path: str = 'data/epicurious_recipes_clean.pq') -> pd.DataFrame:
    """Load the recipes into the file

    Args:
        path (str, optional): the location of the recipes. 
            Defaults to 'data/epicurious_recipes.csv'.
        save_path (str, optional): the location into which the 
            cleaned data should be saved

    Returns:
        pd.DataFrame: the loaded recipes

    """
    df = pd.read_csv(path)

    binary_columns = [
        'pork', 'ground beef', 'radish', 'fontina', 'grapefruit', 'spirit',
        'omelet', 'sesame oil', 'mint', 'stock',
        'orange juice', 'bitters', 'pepper', 'date', 'potato salad',
        'peanut', 'pecan', 'vinegar', 'amaretto', 'cranberry sauce', 'capers',
        'berry', 'lamb', 'plum',
        'marsala', 'poblano', 'quail',
        'jam or jelly', 'tuna', 'veal',
        'steak', 'parsley', 'cashew',
        'goose', 'game', 'flat bread', 'wild rice', 'fortified wine',
        'blackberry', 'vodka', 'macadamia nut', 'bulgur', 'garlic',
        'semolina', 'cognac/armagnac', 'rosemary', 'pernod',
        'bean', 'marscarpone', 'sherry', 'asian pear', 'feta',
        'taco', 'radicchio', 'milk/cream', 'leafy green', 'oregano', 'beer', 'cornmeal',
        'collard greens', 'hominy/cornmeal/masa',
        'saffron', 'tomato', 'paprika', 'noodle', 'sweet potato/yam',
        'poach', 'tofu', 'mozzarella', 'triple sec', 'coconut', 'chive',
        'tropical fruit', 'celery', 'caviar',
        'root vegetable', 'brine', 'apricot', 'buffalo',
        'aperitif', 'rutabaga', 'spinach',
        'iced coffee', 'snapper', 'bass', 'swiss cheese', 'honey', 'tea',
        'fish', 'poppy', 'vanilla',
        'mustard', 'oyster', 'buttermilk', 'pork rib', 'cantaloupe',
        'yogurt', 'macaroni and cheese', 'lychee',
        'fruit juice', 'strawberry', 'clam',
        'chili', 'bell pepper', 'cookies', 'pea',
        'lingonberry', 'plantain', 'cottage cheese',
        'rabbit', 'lentil', 'tree nut', 'skewer',
        'peanut butter', 'crab', 'ham', 'broccoli', 'pistachio', 'fruit',
        'sorbet', 'lime juice', 'nut', 'curry', 'tequila',
        'ground lamb', 'whole wheat', 'watermelon', 'hot pepper',
        'jícama', 'trout', 'grains', 'salmon', 'lime', 'sour cream',
        'kirsch', 'cinnamon', 'green bean',
        'ricotta', 'avocado', 'cherry', 'couscous',
        'ice cream', 'lemon juice', 'molasses',
        'dill', 'pomegranate', 'raisin',
        'persimmon', 'anchovy', 'spice',
        'octopus', 'frangelico', 'lamb chop', 'venison', 'basil', 'waffle',
        'sake', 'white wine', 'cilantro', 'gin', 'granola',
        'salad', 'grappa', 'red wine', 'zucchini', 'quinoa',
        'guava', 'pear', 'asparagus', 'hummus', 'grand marnier',
        'squid',
        'prune', 'kale', 'salad dressing', 'vermouth', 'kahlúa', 'turkey',
        'grape', 'cardamom', 'parsnip', 'quince', 'wasabi', 'dried fruit',
        'marshmallow', 'squash', 'chocolate', 'muffin', 'biscuit', 'champagne',
        'prosciutto', 'tapioca',
        'parmesan', 'lobster',
        'gouda', 'barley', 'salsa', 'pumpkin', 'meatloaf',
        'cranberry', 'banana', 'rice', 'tilapia',
        'liqueur', 'oatmeal',
        'maple syrup', 'kumquat', 'butter', 'burrito', 'papaya',
        'pork chop', 'oat',
        'pasta', 'custard', 'raspberry', 'apple juice', 'green onion/scallion',
        'lettuce', 'coffee', 'clove', 'lemon',
        'anise', 'midori', 'pineapple',
        'carrot', 'jerusalem artichoke', 'corn', 'sardine',
        'thyme', 'brussel sprout', 'cream cheese', 'melon',
        'rack of lamb', 'sangria', 'phyllo/puff pastry dough', 'meatball',
        'mezcal', 'cod', 'sparkling wine', 'brandy',
        'mussel', 'okra', 'pancake', 'pork tenderloin', 'sourdough',
        'cheddar', 'cheese', 'blue cheese', 'beef shank',
        'chickpea', 'shallot', 'beet', 'meat', 'poultry sausage', 'tarragon',
        'nutmeg', 'shrimp', 'cake',
        'rye', 'beef rib', 'cabbage', 'seafood',
        'fennel', 'pine nut', 'chard',
        'tree nut free', 'apple', 'bacon',
        'bran', 'tomatillo',
        'halibut', 'eggplant', 'créme de cacao', 'soy',
        'potato', 'brisket',
        'arugula', 'scotch', 'whiskey',
        'mustard greens', 'hamburger', 'port', 'duck',
        'pomegranate juice', 'chile pepper', 'cucumber', 'kiwi',
        'sesame', 'watercress',
        'horseradish',
        'butterscotch/caramel', 'cauliflower', 'orange', 'wheat/gluten-free',
        'tangerine',
        'honeydew', 'brie', 'swordfish', 'soy sauce', 'nectarine',
        'blueberry', 'ginger', 'monterey jack',
        'yellow squash', 'fig', 'walnut', 'egg',
        'shellfish', 'broccoli rabe',
        'onion', 'rum', 'citrus', 'passion fruit',
        'turnip', 'chartreuse', 'leek',
        'wine', 'bourbon', 'scallop', 'endive', 'sausage',
        'bok choy', 'beef', 'brown rice',
        'chestnut', 'caraway', 'sage',
        'beef tenderloin', 'escarole', 'mango', 'chicken',
        'mayonnaise', 'olive', 'artichoke', 'breadcrumbs',
        'tamarind', 'yuca', 'currant', 'mushroom', 'bread',
        'lamb shank', 'pickles', 'almond', 'cookie',
        'lemongrass', 'peach', 'hazelnut', 'orzo',
        'poultry', 'lima bean', 'butternut squash', 'sugar snap pea', 'cumin',
        'goat cheese', 'tortillas', 'jalapeño',
        'rhubarb', 'campari', 'coriander', 'legume']
    other_columns = ['calories', 'protein', 'fat',
                     'sodium', 'title', 'rating', '#cakeweek']

    for col in binary_columns:
        df[col] = df[col].astype(bool)

    df = df[binary_columns + other_columns]
    df = df.dropna()

    bad_rows = None
    for col in ['calories', 'protein', 'fat', 'sodium']:
        col_max = df[col].quantile(0.99)
        if bad_rows is None:
            bad_rows = df[col] >= col_max
        else:
            bad_rows = np.bitwise_or(bad_rows, df[col] >= col_max)

        df[col] /= col_max

    df = df[np.invert(bad_rows)]
    df.to_parquet(save_path, index=False)

    return df


if __name__ == '__main__':
    df = clean_recipes()
