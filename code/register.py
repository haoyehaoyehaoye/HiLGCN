import world
import dataloader_mine
import model
import utils
from pprint import pprint

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader_mine.Loader(world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader_mine.LastFM()

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}