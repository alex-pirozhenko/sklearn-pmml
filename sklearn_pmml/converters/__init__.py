import sklearn_pmml.pmml as pmml

p = pmml.PMML()
p.version = "4.1"

p.Header = pmml.Header(copyright='copyright here')

p.DataDictionary = pmml.DataDictionary(numberOfFields=1)
p.DataDictionary.DataField.append(pmml.DataField(dataType='double', name='output', optype="continuous"))

tm = pmml.TreeModel(functionName='classification')
tm.MiningSchema = pmml.MiningSchema()
tm.MiningSchema.MiningField.append(pmml.MiningField(name='output', usageType="predicted"))
tm.Node = pmml.Node()
tm.Node.append(pmml.True_())
tm.Node.ScoreDistribution.append(pmml.ScoreDistribution(confidence=0.1, value=1))
tm.Node.ScoreDistribution.append(pmml.ScoreDistribution(confidence=0.1, value=2))

p.TreeModel.append(tm)
print p.toxml()


