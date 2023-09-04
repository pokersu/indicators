

# series_from(feature_string, _close, _high, _low, _hlc3, f_paramA, f_paramB) =>
#     switch feature_string
#         "RSI" => ml.n_rsi(_close, f_paramA, f_paramB)
#         "WT" => ml.n_wt(_hlc3, f_paramA, f_paramB)
#         "CCI" => ml.n_cci(_close, f_paramA, f_paramB)
#         "ADX" => ml.n_adx(_high, _low, _close, f_paramA)

def series_from():
    pass

# get_lorentzian_distance(int i, int featureCount, FeatureSeries featureSeries, FeatureArrays featureArrays) =>
#     switch featureCount
#         5 => math.log(1+math.abs(featureSeries.f1 - array.get(featureArrays.f1, i))) +
#              math.log(1+math.abs(featureSeries.f2 - array.get(featureArrays.f2, i))) +
#              math.log(1+math.abs(featureSeries.f3 - array.get(featureArrays.f3, i))) +
#              math.log(1+math.abs(featureSeries.f4 - array.get(featureArrays.f4, i))) +
#              math.log(1+math.abs(featureSeries.f5 - array.get(featureArrays.f5, i)))
#         4 => math.log(1+math.abs(featureSeries.f1 - array.get(featureArrays.f1, i))) +
#              math.log(1+math.abs(featureSeries.f2 - array.get(featureArrays.f2, i))) +
#              math.log(1+math.abs(featureSeries.f3 - array.get(featureArrays.f3, i))) +
#              math.log(1+math.abs(featureSeries.f4 - array.get(featureArrays.f4, i)))
#         3 => math.log(1+math.abs(featureSeries.f1 - array.get(featureArrays.f1, i))) +
#              math.log(1+math.abs(featureSeries.f2 - array.get(featureArrays.f2, i))) +
#              math.log(1+math.abs(featureSeries.f3 - array.get(featureArrays.f3, i)))
#         2 => math.log(1+math.abs(featureSeries.f1 - array.get(featureArrays.f1, i))) +
#              math.log(1+math.abs(featureSeries.f2 - array.get(featureArrays.f2, i)))

def get_lorentzian_distance(i: int, ):
    pass

def filters():
    pass

def predicate():
    pass