import init_tests
import pytest
from logistic_regression import *
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

class TestLogisticRegression:
    class TestTwoClasses:
        @pytest.fixture
        def data(self):
            return datasets.make_classification(500, n_features=1, n_informative=1, n_redundant=0, n_clusters_per_class=1, random_state=10)

        @pytest.fixture
        def x(self, data):
            return data[0]

        @pytest.fixture
        def y(self, data):
            return data[1]

        class TestBinomial:
            @pytest.fixture
            def subject(self, x, y):
                return Binomial()

            def test_has_a_decent_accuracy(self, subject, x, y):
                subject.fit(x, y)

                assert np.sum(subject.predict(x) == y) >= 400

        class TestSkLearn:
            @pytest.fixture
            def subject(self):
                return LogisticRegression()

            def test_has_a_decent_accuracy(self, subject, x, y):
                subject.fit(x, y)

                assert np.sum(subject.predict(x) == y) >= 400


    class TestMultipleClasses:
        @pytest.fixture
        def data(self):
            return datasets.make_classification(500, n_features=4, n_informative=4, n_classes=4, n_redundant=0, n_clusters_per_class=1)

        @pytest.fixture
        def x(self, data):
            return data[0]

        @pytest.fixture
        def y(self, data):
            return data[1]

        class TestMultinomial:
            @pytest.fixture
            def subject(self, x, y):
                return Multinomial()

            def test_has_a_decent_accuracy(self, subject, x, y):
                subject.fit(x, y)

                assert np.sum(subject.predict(x) == y) >= 300

        class TestSkLearn:
            @pytest.fixture
            def subject(self):
                return LogisticRegression()

            def test_has_a_decent_accuracy(self, subject, x, y):
                subject.fit(x, y)

                assert np.sum(subject.predict(x) == y) >= 400
