import dataManager
from unittest import TestCase

location = "/home/pooja/PycharmProjects/homeCredit/baseDatasets/"  # chnge\
outputLocation= "/home/pooja/PycharmProjects/homeCredit/tests/"
class dataObject(TestCase):
    def test(self):
        dataset = dataManager.dataObject(loc=location, name='application_test', primaryKey=None, rollupKey=None)
        dataset.load()
        dataset.save(loc=location)
        self.assertTrue(all([dataset.df.shape[0] == 150]), msg='may Be dataset object is not formed')

class dataOwner(TestCase):

    def test(self):
        dataset = dataManager.dataOwner(loc='/home/pooja/PycharmProjects/homeCredit/tests/dataManagerFiles/')
        dataset.addDatacards("/home/pooja/PycharmProjects/homeCredit/baseDatasets/")
        dataset.save() #works
        c=dataset.fetchdatacards()

        self.assertTrue(all([len(dataset.cards) == 7]), msg='may Be dataset object is not formed')







