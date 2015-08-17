package sklearn.pmml.jpmml;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

import com.google.common.collect.Sets;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.IOUtil;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.Evaluator;
import org.jpmml.evaluator.ModelEvaluator;
import org.jpmml.evaluator.ModelEvaluatorFactory;
import org.jpmml.manager.PMMLManager;
import org.supercsv.io.CsvMapReader;
import org.supercsv.io.CsvMapWriter;
import org.supercsv.prefs.CsvPreference;
import org.xml.sax.SAXException;

import javax.xml.bind.JAXBException;

import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Created by evancox on 7/23/15.
 */
public class JPMMLCSVEvaluator
{
    private static final Logger logger = Logger.getLogger(JPMMLCSVEvaluator.class.getCanonicalName());

    static PMML pmmlFromXml(final InputStream is)
    {
        try
        {
            return IOUtil.unmarshal(is);
        }
        catch (SAXException | JAXBException e)
        {
            throw new RuntimeException("Error reading PMML.", e);
        }
    }

    static Evaluator evaluatorFromPmml(final PMML pmml)
    {
        final PMMLManager pmmlManager = new PMMLManager(pmml);
        final ModelEvaluator<?> modelEvaluator = (ModelEvaluator<?>) pmmlManager.getModelManager(null, ModelEvaluatorFactory.getInstance());

        return modelEvaluator;
    }

    static Evaluator evaluatorFromXml(final InputStream is)
    {
        // Adapted from:
        //   * https://github.com/jpmml/jpmml/blob/master/README.md
        //   * https://github.com/jpmml/jpmml-example/blob/master/src/main/java/org/jpmml/example/CsvEvaluationExample.java
        return evaluatorFromPmml(pmmlFromXml(is));
    }

    static List<Map<FieldName, ?>> getPredictions(Evaluator evaluator, String csvFeaturesFile) throws IOException
    {
        try (final CsvMapReader csvMapReader = new CsvMapReader(new FileReader(csvFeaturesFile), CsvPreference.STANDARD_PREFERENCE)) {
            final String[] headers = csvMapReader.getHeader(true);
            final Map<String, FieldName> fieldNameMap = new HashMap<>(headers.length);
            for (String header : Arrays.asList(headers))
            {
                fieldNameMap.put(header, new FieldName(header));
            }

            Map<String, String> rawCsvMap;
            final List<Map<FieldName, ?>> predictions = Lists.newArrayList();
            while ((rawCsvMap = csvMapReader.read(headers)) != null) {
                final Map<FieldName, FieldValue> featureMap = Maps.newHashMapWithExpectedSize(rawCsvMap.size());
                for (Map.Entry<String, String> keyValue : rawCsvMap.entrySet())
                {
                    final FieldName fieldName = fieldNameMap.get(keyValue.getKey());
                    final FieldValue fieldValue = evaluator.prepare(fieldName, keyValue.getValue());
                    featureMap.put(fieldName, fieldValue);
                }
                predictions.add(evaluator.evaluate(featureMap));
            }
            return predictions;
        }
    }

    static void writePredictions(Evaluator evaluator, List<Map<FieldName, ?>> predictions, String outputFile) throws IOException
    {
        final int outputFieldCount = evaluator.getOutputFields().size();
        final Set<FieldName> outputFields = Sets.newHashSetWithExpectedSize(outputFieldCount);
        final String[] header = new String[outputFieldCount];
        int index = 0;
        for (FieldName fieldName : evaluator.getOutputFields())
        {
            outputFields.add(fieldName);
            header[index++] = fieldName.toString();
        }

        try (final CsvMapWriter csvMapWriter = new CsvMapWriter(new FileWriter(outputFile), CsvPreference.STANDARD_PREFERENCE))
        {
            csvMapWriter.writeHeader(header);
            for (Map<FieldName, ?> prediction : predictions) {

                final Map<String, Object> row = Maps.newHashMapWithExpectedSize(prediction.size());
                for (Map.Entry<FieldName, ?> keyValue : prediction.entrySet())
                {
                    row.put(keyValue.getKey().toString(), keyValue.getValue());
                }
                csvMapWriter.write(row, header);
            }
        }
    }

    public static void main(String[] args)
    {
        if (args.length != 3)
        {
            throw new RuntimeException("Expected PMML file, feature data, and output predictions file");
        }
        final String pmmlFile = args[0];
        final String csvFeaturesFile = args[1];
        final String outputFile = args[2];
        try
        {
            Evaluator evaluator = evaluatorFromXml(new FileInputStream(pmmlFile));
            final List<Map<FieldName, ?>> predictions = getPredictions(evaluator, csvFeaturesFile);
            writePredictions(evaluator, predictions, outputFile);
            logger.info(String.format("Wrote %d predictions from %s to %s", predictions.size(), csvFeaturesFile, outputFile));
        }
        catch (IOException ex)
        {
            logger.log(Level.SEVERE, "IOException", ex);
            System.exit(1);
        }


    }

}
