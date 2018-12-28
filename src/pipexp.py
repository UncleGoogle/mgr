import json
import sys


fname = sys.argv[1]

with open(fname, 'r') as f:
    data = json.load(f)
if not data:
    data = {}

data_serie = input('data serie... ')
if not data_serie:
    sys.exit('no serie defined')

data[data_serie] = data.get(data_serie, {})

focus_reference = input('focus_reference...(insert nothing to skip) ')
if focus_reference:
    data[data_serie]['focus_reference'] = str(focus_reference)

new_imgs = []
while True:
    try:
        img = {}
        img['x'] = int(input('x...'))
        img['kind'] = input('kind...')
        img['t'] = int(input('t...'))
        img['name'] = int(input('photo number...'))
    except KeyboardInterrupt:
        print(f'\nExiting. {len(new_imgs)} images was added')
        break
    except ValueError as e:
        print(e, 'repeating last input data')
        continue
    else:
        new_imgs.append(img)

imgs = data[data_serie].get('imgs', [])
data[data_serie]['imgs'] = imgs + new_imgs
print('saving data...')
try:
    with open(fname, 'w') as f:
        json.dump(data, f, indent=4)
except Exception as e:
    print(e)
