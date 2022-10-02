template <class T>
struct mt_ad_type{
  typedef metricTensorDummy type;
};

template<>
struct mt_ad_type<metricTensorSparse >{
  typedef metricTensorSparseNumeric<stan::math::var> type;
};

template<>
struct mt_ad_type<metricTensorDense>{
  typedef metricTensorDenseNumeric<stan::math::var> type;
};
