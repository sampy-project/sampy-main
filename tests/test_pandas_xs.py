import sampy
import numpy as np
import unittest
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

sampy.use_debug_mode(sampy.DataFrameXS)


class TestSetItemsPandasXS(unittest.TestCase):
    def setUp(self):
        self.df = sampy.DataFrameXS()

    def test_key_empty(self):
        with self.assertRaises(ValueError) as cm:
            self.df[None] = None
        self.assertEqual(str(cm.exception), "A column name should be a string in a dataframe_xs.")

    def test_key_int(self):
        with self.assertRaises(ValueError) as cm:
            self.df[1] = None
        self.assertEqual(str(cm.exception), "A column name should be a string in a dataframe_xs.")

    def test_key_float(self):
        with self.assertRaises(ValueError) as cm:
            self.df[1.] = None
        self.assertEqual(str(cm.exception), "A column name should be a string in a dataframe_xs.")

    def test_key_list(self):
        with self.assertRaises(ValueError) as cm:
            self.df[[]] = None
        self.assertEqual(str(cm.exception), "A column name should be a string in a dataframe_xs.")

    def test_key_dict(self):
        with self.assertRaises(ValueError) as cm:
            self.df[{}] = None
        self.assertEqual(str(cm.exception), "A column name should be a string in a dataframe_xs.")

    def test_key_array_of_a_single_str(self):
        with self.assertRaises(ValueError) as cm:
            self.df[np.array('abc')] = None
        self.assertEqual(str(cm.exception), "A column name should be a string in a dataframe_xs.")

    def test_add_first_column_as_empty_list(self):
        self.df['1'] = []
        self.assertEqual(self.df.shape, (0, 1))
        self.assertIsNone(self.df.list_col[0])

    def test_add_first_column_as_list(self):
        self.df['test'] = [1, 2, 3, 4]
        with self.subTest():
            self.assertEqual(len(self.df.list_col), 1)

        with self.subTest():
            self.assertEqual(self.df.nb_cols, 1)

        with self.subTest():
            self.assertEqual(self.df.nb_rows, 4)

        with self.subTest():
            self.assertTrue( 'test' in self.df.dict_colname_to_index )
            self.assertEqual(self.df.dict_colname_to_index['test'], 0)
            self.assertTrue((self.df.list_col[0] == np.array([1, 2, 3, 4])).all())

        with self.subTest():
            self.assertEqual(str(self.df.list_col[0].dtype), 'int32')

    def test_add_first_column_as_array(self):
        self.df['test'] = np.array([1, 2, 3, 4])

        with self.subTest():
            self.assertEqual(len(self.df.list_col), 1)

        with self.subTest():
            self.assertEqual(self.df.nb_cols, 1)

        with self.subTest():
            self.assertEqual(self.df.nb_rows, 4)

        with self.subTest():
            self.assertTrue( 'test' in self.df.dict_colname_to_index )
            self.assertEqual(self.df.dict_colname_to_index['test'], 0)
            self.assertTrue((self.df.list_col[0] == np.array([1, 2, 3, 4])).all())

        with self.subTest():
            self.assertEqual(str(self.df.list_col[0].dtype), 'int32')

    def test_value_is_a_dict(self):
        with self.assertRaises(ValueError) as cm:
            self.df['test'] = {}
        self.assertEqual(str(cm.exception), "The proposed column is of an unsupported type. Supported types are " +
                         str(self.df.LIST_ALLOWED_TYPE))

    def test_value_is_list_of_string(self):
        with self.assertRaises(ValueError) as cm:
            self.df['test'] = ['abc', 'def']
        self.assertEqual(str(cm.exception), "The proposed column is of an unsupported type. Supported types are " +
                         str(self.df.LIST_ALLOWED_TYPE))

    def test_value_is_arr_complex(self):
        with self.assertRaises(ValueError) as cm:
            self.df['test'] = np.array([1, 2, 3, 4, 5], dtype=np.complex_)
        self.assertEqual(str(cm.exception), "The proposed column is of an unsupported type. Supported types are " +
                         str(self.df.LIST_ALLOWED_TYPE))

    def test_add_wrong_col_on_non_empty_df(self):
        self.df['1'] = [1, 2]
        with self.subTest():
            with self.assertRaises(ValueError) as cm:
                self.df['test'] = ['abc', 'def']
            self.assertEqual(str(cm.exception), "The proposed column is of an unsupported type. Supported types are " +
                             str(self.df.LIST_ALLOWED_TYPE))
        with self.subTest():
            with self.assertRaises(ValueError) as cm:
                self.df['test'] = np.array([1, 2], dtype=np.complex_)
            self.assertEqual(str(cm.exception), "The proposed column is of an unsupported type. Supported types are " +
                             str(self.df.LIST_ALLOWED_TYPE))
        with self.subTest():
            with self.assertRaises(ValueError) as cm:
                self.df['test'] = {}
            self.assertEqual(str(cm.exception), "The proposed column is of an unsupported type. Supported types are " +
                             str(self.df.LIST_ALLOWED_TYPE))

    def test_add_new_column_of_wrong_length(self):
        self.df['1'] = [1, 2, 3, 4]
        with self.subTest():
            with self.assertRaises(ValueError) as cm:
                self.df['2'] = [1, 2, 3]
            self.assertEqual(str(cm.exception), "Dataframe has " + str(self.df.nb_rows) + " rows while the input has 3.")
        with self.subTest():
            with self.assertRaises(ValueError) as cm:
                self.df['2'] = [1, 2, 3, 4, 5]
            self.assertEqual(str(cm.exception),
                             "Dataframe has " + str(self.df.nb_rows) + " rows while the input has 5.")

    def test_add_empty_column(self):
        self.df['1'] = [1, 2, 3, 4]
        self.df['test'] = None
        with self.subTest():
            self.assertEqual(self.df.nb_rows, 4)
        with self.subTest():
            self.assertEqual(self.df.nb_cols, 2)
        with self.subTest():
            self.assertIn('test', self.df.dict_colname_to_index)
            self.assertEqual(self.df.dict_colname_to_index['test'], 1)
            self.assertIsNone(self.df.list_col[1])

    def test_add_column_by_single_value(self):
        self.df['1'] = [1, 2, 3, 4]
        self.df['2'] = True
        self.assertEqual(self.df.list_col[self.df.dict_colname_to_index['2']].shape, (4,))
        self.assertEqual(str(self.df.list_col[self.df.dict_colname_to_index['2']].dtype), 'bool')
        self.assertTrue(self.df.list_col[1].all())

    def test_add_column_as_empty_list(self):
        self.df['1'] = [1, 2, 3, 4]
        self.df['2'] = []
        self.assertEqual(self.df.shape, (4, 2))
        self.assertIsNone(self.df.list_col[1])

    def test_add_column_as_empty_tuple(self):
        self.df['1'] = [1, 2, 3, 4]
        self.df['2'] = tuple()
        self.assertEqual(self.df.shape, (4, 2))
        self.assertIsNone(self.df.list_col[1])

    def test_add_column_as_list_of_int(self):
        self.df['1'] = [1, 2, 3]
        self.df['2'] = [4, 5, 6]
        self.assertEqual(self.df.shape, (3, 2))
        self.assertTrue((self.df.list_col[self.df.dict_colname_to_index['2']] == np.array([4, 5, 6])).all())

    def test_is_empty(self):
        self.assertTrue(self.df.is_empty)
        self.df['1'] = None
        self.assertTrue(self.df.is_empty)
        self.df['2'] = [1, 2, 3]
        self.assertFalse(self.df.is_empty)
        self.df['2'] = None
        self.assertTrue(self.df.is_empty)

    def test_shape(self):
        self.assertEqual(self.df.shape, (0, 0))
        self.df['1'] = None
        self.assertEqual(self.df.shape, (0, 1))
        self.df['2'] = [1, 2, 3]
        self.assertEqual(self.df.shape, (3, 2))
        self.df['2'] = None
        self.assertEqual(self.df.shape, (0, 2))


class TestGetItemsPandasXS(unittest.TestCase):
    def setUp(self):
        self.df = sampy.DataFrameXS()
        self.df['col1'] = [1, 2, 3, 4]
        self.df['col2'] = [2., 3., 4., 5.]

    def test_check_cols_are_retrieved_correctly(self):
        self.assertTrue((self.df['col1'] == np.array([1, 2, 3, 4])).all())
        self.assertTrue((self.df['col2'] == np.array([2., 3., 4., 5.])).all())
        self.assertFalse((self.df['col1'] == self.df['col2']).all())
        self.assertTrue(str(self.df['col1'].dtype).startswith('int'))

    def test_getitems_returns_ref(self):
        self.df['col1'][0] = 0
        self.assertTrue((self.df['col1'] == np.array([0, 2, 3, 4])).all())

    def test_bool_arr_indexing(self):
        arr_bool = np.array([True, False, True, False])
        df = self.df[arr_bool]
        self.assertEqual(df.nb_rows, 2)
        self.assertEqual(df.nb_cols, 2)
        self.assertEqual(df.list_col_name, self.df.list_col_name)
        self.assertTrue((df['col1'] == np.array([1, 3])).all())
        self.assertTrue((df['col2'] == np.array([2., 4.])).all())

    def test_int_arr_indexing(self):
        arr_int = np.array([2, 0, 1, 3])
        df = self.df[arr_int]
        self.assertEqual(df.nb_rows, 4)
        self.assertEqual(df.nb_cols, 2)
        self.assertEqual(df.list_col_name, self.df.list_col_name)
        self.assertTrue((df['col1'] == np.array([3, 1, 2, 4])).all())
        self.assertTrue((df['col2'] == np.array([4., 2., 3., 5.])).all())
        self.assertFalse((df['col1'] == df['col2']).all())
        self.assertTrue(str(df['col1'].dtype).startswith('int'))

    def test_wrong_indexing(self):
        with self.subTest():
            with self.assertRaises(TypeError) as cm:
                self.df[{}]
        with self.subTest():
            with self.assertRaises(TypeError) as cm:
                self.df[1]
        with self.subTest():
            with self.assertRaises(TypeError) as cm:
                self.df[[True, True, False]]
        with self.subTest():
            with self.assertRaises(TypeError) as cm:
                self.df[np.array([1., 2., 3., 4.])]
        with self.subTest():
            with self.assertRaises(KeyError) as cm:
                self.df['wrong_col_name']
        with self.subTest():
            with self.assertRaises(IndexError) as cm:
                self.df[np.array([True, True, False])]
        with self.subTest():
            with self.assertRaises(IndexError) as cm:
                self.df[np.array([True, True, False, False, False])]
        with self.subTest():
            with self.assertRaises(IndexError) as cm:
                self.df[np.array([0, 5])]


class TestGenericMethodsPandasXS(unittest.TestCase):
    def setUp(self):
        self.df = sampy.DataFrameXS()
        self.df['col1'] = [1, 2, 3, 4]
        self.df['col2'] = [2., 3., 4., 5.]

    def test_scramble(self):
        with self.assertRaises(ValueError):
            self.df.scramble(permutation=np.array([2, 0, 1]))
        with self.assertRaises(TypeError):
            self.df.scramble(permutation=[2, 0, 1, 3])
        with self.assertRaises(ValueError):
            self.df.scramble(permutation=np.array([2, 0, 1, 4]))
        with self.assertRaises(ValueError):
            self.df.scramble(permutation=np.array([[2, 0, 1, 4]]))
        with self.assertRaises(TypeError):
            self.df.scramble(permutation=np.array([]))
        with self.assertRaises(ValueError):
            self.df.scramble(permutation=np.array(0))
        perm = np.array([2, 0, 1, 3])
        self.df.scramble(permutation=perm)
        self.assertEqual(self.df.nb_rows, 4)
        self.assertEqual(self.df.nb_cols, 2)
        self.assertEqual(self.df.list_col_name, ['col1', 'col2'])
        self.assertTrue((self.df['col1'] == np.array([3, 1, 2, 4])).all())
        self.assertTrue((self.df['col2'] == np.array([4., 2., 3., 5.])).all())
        self.assertFalse((self.df['col1'] == self.df['col2']).all())
        self.assertTrue(str(self.df['col1'].dtype).startswith('int'))

    def test_get_copy(self):
        with self.assertRaises(KeyError):
            self.df.get_copy('wrong_col_name')

        with self.assertRaises(TypeError):
            self.df.get_copy([])

        with self.assertRaises(TypeError):
            self.df.get_copy(1)

        u = self.df.get_copy('col1')
        self.assertTrue((u == self.df['col1']).all())
        u[0] = 0
        self.assertFalse((u == self.df['col1']).all())

    def test_check_arr_in_col(self):
        with self.subTest():
            with self.assertRaises(ValueError):
                input_arr = np.array([1., 2., 3., 4.])
                self.df.check_arr_in_col(input_arr, 'col1')
        with self.subTest():
            with self.assertRaises(ValueError):
                input_arr = np.array([1, 2, 3, 4])
                self.df.check_arr_in_col(input_arr, 'col2')
        with self.subTest():
            with self.assertRaises(ValueError):
                input_arr = np.array([[1., 2.], [3., 4.]])
                self.df.check_arr_in_col(input_arr, 'col2')
        with self.subTest():
            with self.assertRaises(ValueError):
                input_arr = np.array(2.)
                self.df.check_arr_in_col(input_arr, 'col2')
        with self.subTest():
            with self.assertRaises(ValueError):
                input_arr = np.array(2.)
                self.df.check_arr_in_col(input_arr, 'col2')
        with self.subTest():
            input_arr = np.array([3, 4, 8, 0, 10, 1])
            input_arr = input_arr.astype('int8')
            with self.assertRaises(ValueError):
                result = self.df.check_arr_in_col(input_arr, 'col1')
        with self.subTest():
            input_arr = np.array([3, 4, 8, 0, 10, 1])
            result = self.df.check_arr_in_col(input_arr, 'col1')
            self.assertTrue((result == np.array([True, True, False, False, False, True])).all())

    def test_check_arr_in_col_conditional(self):
        with self.subTest():
            with self.assertRaises(ValueError):
                condition = np.array([True, True, False, False])
                input_arr = np.array([1., 2., 3., 4.])
                self.df.check_arr_in_col(input_arr, 'col1', condition=condition)
        with self.subTest():
            with self.assertRaises(ValueError):
                condition = np.array([True, True, False, False])
                input_arr = np.array([1, 2, 3, 4])
                self.df.check_arr_in_col(input_arr, 'col2', condition=condition)
        with self.subTest():
            with self.assertRaises(ValueError):
                condition = np.array([[True, True], [False, False]])
                input_arr = np.array([[1., 2.], [3., 4.]])
                self.df.check_arr_in_col(input_arr, 'col2', condition=condition)
        with self.subTest():
            with self.assertRaises(ValueError):
                condition = np.array([True])
                input_arr = np.array(2.)
                self.df.check_arr_in_col(input_arr, 'col2', condition=condition)
        with self.subTest():
            with self.assertRaises(ValueError):
                condition = np.array([True])
                input_arr = np.array({})
                self.df.check_arr_in_col(input_arr, 'col2', condition=condition)
        with self.subTest():
            condition = np.array([True, False, False, True, True, True])
            input_arr = np.array([3, 4, 8, 0, 10, 1])
            input_arr = input_arr.astype('int8')
            with self.assertRaises(ValueError):
                self.df.check_arr_in_col(input_arr, 'col1', condition=condition)
        with self.subTest():
            condition = np.array([True, False, False, True, True, True])
            input_arr = np.array([3, 4, 8, 0, 10, 1])
            result = self.df.check_arr_in_col(input_arr, 'col1', condition=condition)
            self.assertTrue((result == np.array([True, False, False, False, False, True])).all())
        with self.subTest():
            condition = np.array([True, False, False, True, True])
            input_arr = np.array([3, 4, 8, 0, 10, 1])
            with self.assertRaises(ValueError):
                self.df.check_arr_in_col(input_arr, 'col1', condition=condition)
        with self.subTest():
            condition = np.array([1, 0, 0, 1, 1, 1])
            input_arr = np.array([3, 4, 8, 0, 10, 1])
            with self.assertRaises(ValueError):
                self.df.check_arr_in_col(input_arr, 'col1', condition=condition)
        with self.subTest():
            condition = {}
            input_arr = np.array([3, 4, 8, 0, 10, 1])
            with self.assertRaises(ValueError):
                self.df.check_arr_in_col(input_arr, 'col1', condition=condition)
        with self.subTest():
            condition = np.array([[True, False], [False, True], [True, True]])
            input_arr = np.array([3, 4, 8, 0, 10, 1])
            with self.assertRaises(ValueError):
                self.df.check_arr_in_col(input_arr, 'col1', condition=condition)

    def test_concat_inplace_is_false(self):
        with self.subTest():
            df = sampy.DataFrameXS()
            df['col1'] = [5, 6, 7, 8]
            df['col2'] = [6., 7., 8., 9.]
            c_df = self.df.concat(df, inplace=False)

            # concats happened as expected
            self.assertTrue((c_df['col1'] == np.array([1, 2, 3, 4, 5, 6, 7, 8])).all())
            self.assertTrue((c_df['col2'] == np.array([2., 3., 4., 5., 6., 7., 8., 9.])).all())

            # columns are new copied columns
            c_df['col1'][0] = 0
            self.assertFalse(c_df['col1'][0] == self.df['col1'][0])

        with self.subTest():
            # same test as the previous one, but there is an extra column
            df = sampy.DataFrameXS()
            df['col1'] = [5, 6, 7, 8]
            df['col2'] = [6., 7., 8., 9.]
            df['col3'] = [1, 2, 3, 4]
            c_df = self.df.concat(df, inplace=False)

            # check that there is no col3 in the dataframe
            with self.assertRaises(KeyError):
                c_df['col3']

            # perform all the previous checks in this case.
            # concat happened as expected
            self.assertTrue((c_df['col1'] == np.array([1, 2, 3, 4, 5, 6, 7, 8])).all())
            self.assertTrue((c_df['col2'] == np.array([2., 3., 4., 5., 6., 7., 8., 9.])).all())

            # columns are new copied columns
            c_df['col1'][0] = 0
            self.assertFalse(c_df['col1'][0] == self.df['col1'][0])

        # check that an error is raised when the df don't have the right col_names
        with self.subTest():
            df = sampy.DataFrameXS()
            df['col1'] = [5, 6, 7, 8]
            df['col3'] = [6., 7., 8., 9.]
            with self.assertRaises(KeyError):
                self.df.concat(df, inplace=False)

        # check that appending with an empty dataframe returns a copy
        with self.subTest():
            df = sampy.DataFrameXS()
            c_df = self.df.concat(df, inplace=False)
            self.assertTrue((c_df['col1'] == self.df['col1']).all())
            self.assertTrue((c_df['col2'] == self.df['col2']).all())

            c_df['col1'][0] = 0
            self.assertFalse(c_df['col1'][0] == self.df['col1'][0])

        # check that having a null column in the input is correctly dealt with
        with self.subTest():
            df = sampy.DataFrameXS()
            df['col1'] = [5, 6, 7, 8]
            df['col2'] = None
            c_df = self.df.concat(df, inplace=False)
            self.assertTrue((c_df['col1'] == np.array([1, 2, 3, 4, 5, 6, 7, 8])).all())
            self.assertTrue((np.nan_to_num(c_df['col2'], nan=0.) == np.array([2., 3., 4., 5., 0., 0., 0., 0.])).all())
            self.assertEqual(str(c_df['col2'].dtype), str(self.df['col2'].dtype))

        # same thing but in reverse
        with self.subTest():
            df_in = sampy.DataFrameXS()
            df_in['col1'] = [1, 2, 3, 4]
            df_in['col2'] = None

            df = sampy.DataFrameXS()
            df['col1'] = [5, 6, 7, 8]
            df['col2'] = [6., 7., 8., 9.]
            c_df = df_in.concat(df, inplace=False)
            self.assertTrue((c_df['col1'] == np.array([1, 2, 3, 4, 5, 6, 7, 8])).all())
            self.assertTrue((np.nan_to_num(c_df['col2'], 0.) == np.array([0., 0., 0., 0., 6., 7., 8., 9.])).all())
            self.assertEqual(str(c_df['col2'].dtype), str(df['col2'].dtype))

        # When the first df is empty
        with self.subTest():
            df_in = sampy.DataFrameXS()
            df_in['col1'] = None
            df_in['col2'] = None

            df = sampy.DataFrameXS()
            df['col1'] = [5, 6, 7, 8]
            df['col2'] = [6., 7., 8., 9.]
            c_df = df_in.concat(df, inplace=False)

            self.assertTrue((c_df['col1'] == np.array([5, 6, 7, 8])).all())
            self.assertEqual(str(c_df['col1'].dtype), str(df['col1'].dtype))
            self.assertTrue((np.nan_to_num(c_df['col2'], nan=0.) == np.array([6., 7., 8., 9.])).all())
            self.assertEqual(str(c_df['col2'].dtype), str(df['col2'].dtype))

            # check that a copy is retrieved
            c_df['col1'][0] = 0
            self.assertFalse((c_df['col1'] == df['col1']).all())

        # adding an empty column on an empty column
        with self.subTest():
            df_in = sampy.DataFrameXS()
            df_in['col1'] = [1, 2, 3, 4]
            df_in['col2'] = None

            df = sampy.DataFrameXS()
            df['col1'] = [5, 6, 7, 8]
            df['col2'] = None

            c_df = df_in.concat(df, inplace=False)

            self.assertIsNone(c_df['col2'])
            self.assertTrue((c_df['col1'] == np.array([1, 2, 3, 4, 5, 6, 7, 8])).all())
            self.assertEqual(str(c_df['col1'].dtype), str(df['col1'].dtype))

    def test_concat_inplace_is_true_basic(self):
        df = sampy.DataFrameXS()
        df['col1'] = [5, 6, 7, 8]
        df['col2'] = [6., 7., 8., 9.]
        self.df.concat(df, inplace=True)

        # concats happened as expected
        self.assertTrue((self.df['col1'] == np.array([1, 2, 3, 4, 5, 6, 7, 8])).all())
        self.assertTrue((self.df['col2'] == np.array([2., 3., 4., 5., 6., 7., 8., 9.])).all())

    def test_concat_inplace_is_true_extra_col(self):
        df = sampy.DataFrameXS()
        df['col1'] = [5, 6, 7, 8]
        df['col2'] = [6., 7., 8., 9.]
        df['col3'] = [1, 2, 3, 4]
        self.df.concat(df, inplace=True)

        # check that there is no col3 in the dataframe
        with self.assertRaises(KeyError):
            self.df['col3']

        # perform all the previous checks in this case.
        # concat happened as expected
        self.assertTrue((self.df['col1'] == np.array([1, 2, 3, 4, 5, 6, 7, 8])).all())
        self.assertTrue((self.df['col2'] == np.array([2., 3., 4., 5., 6., 7., 8., 9.])).all())
        self.assertEqual(self.df.nb_rows, 8)

    def test_concat_inplace_is_true_wrong_col_name(self):
        df = sampy.DataFrameXS()
        df['col1'] = [5, 6, 7, 8]
        df['col3'] = [6., 7., 8., 9.]
        with self.assertRaises(KeyError):
            self.df.concat(df, inplace=True)

    def test_concat_inplace_is_true_empty_col_on_empty_col(self):
        self.df['col2'] = None

        df = sampy.DataFrameXS()
        df['col1'] = [5, 6, 7, 8]
        df['col2'] = None

        self.df.concat(df, inplace=True)

        self.assertIsNone(self.df['col2'])
        self.assertTrue((self.df['col1'] == np.array([1, 2, 3, 4, 5, 6, 7, 8])).all())
        self.assertEqual(str(self.df['col1'].dtype), str(df['col1'].dtype))

    def test_concat_inplace_is_true_empty_col_in_input(self):
        df = sampy.DataFrameXS()
        df['col1'] = [5, 6, 7, 8]
        df['col2'] = None

        self.df.concat(df, inplace=True)

        self.assertTrue((np.nan_to_num(self.df['col2'], nan=0.) == np.array([2., 3., 4., 5., 0., 0., 0., 0.])).all())
        self.assertTrue((self.df['col1'] == np.array([1, 2, 3, 4, 5, 6, 7, 8])).all())
        self.assertEqual(str(self.df['col1'].dtype), str(df['col1'].dtype))

    def test_concat_inplace_is_true_on_empty_df(self):
        self.df['col1'] = None
        self.df['col2'] = None

        df = sampy.DataFrameXS()
        df['col1'] = [5, 6, 7, 8]
        df['col2'] = [6., 7., 8., 9.]
        self.df.concat(df, inplace=True)

        self.assertTrue((self.df['col1'] == np.array([5, 6, 7, 8])).all())
        self.assertEqual(str(self.df['col1'].dtype), str(df['col1'].dtype))
        self.assertTrue((np.nan_to_num(self.df['col2'], nan=0.) == np.array([6., 7., 8., 9.])).all())
        self.assertEqual(str(self.df['col2'].dtype), str(df['col2'].dtype))

        self.assertEqual(self.df.nb_rows, 4)


if __name__ == '__main__':
    unittest.main()
