from unittest import TestCase
import os
import projectManager
outputLocation="/home/pooja/PycharmProjects/homeCredit/tests/"
class TestprojectManager(TestCase):
    def test_create_folder(self):
        test1= projectManager.projectOwner(outputLocation)
        test1.initializeFolders()
        self.assertTrue(all([os.path.exists(test1.loc+'/projectMangerFiles')]))


