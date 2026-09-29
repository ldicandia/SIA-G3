use thiserror::Error;

use super::data::DIGIT_CLASSES;

#[derive(Clone, Debug)]
pub struct ClassificationMetrics {
    pub accuracy: f64,
    pub confusion: [[usize; DIGIT_CLASSES]; DIGIT_CLASSES],
    pub per_class_recall: [Option<f64>; DIGIT_CLASSES],
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ClassificationError {
    #[error("predictions and labels must have the same non-zero length")]
    InvalidLengths,
    #[error("class values must be in 0..10")]
    InvalidClass,
}

pub fn classification_metrics(
    predictions: &[usize],
    labels: &[usize],
) -> Result<ClassificationMetrics, ClassificationError> {
    if predictions.is_empty() || predictions.len() != labels.len() {
        return Err(ClassificationError::InvalidLengths);
    }
    if predictions
        .iter()
        .chain(labels)
        .any(|&value| value >= DIGIT_CLASSES)
    {
        return Err(ClassificationError::InvalidClass);
    }
    let mut confusion = [[0; DIGIT_CLASSES]; DIGIT_CLASSES];
    let mut correct = 0;
    for (&prediction, &label) in predictions.iter().zip(labels) {
        confusion[label][prediction] += 1;
        correct += usize::from(prediction == label);
    }
    let per_class_recall = std::array::from_fn(|class| {
        let support = confusion[class].iter().sum::<usize>();
        (support > 0).then(|| confusion[class][class] as f64 / support as f64)
    });
    Ok(ClassificationMetrics {
        accuracy: correct as f64 / labels.len() as f64,
        confusion,
        per_class_recall,
    })
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn multiclass_accuracy_and_recall_match_counts() {
        let metrics = classification_metrics(&[0, 1, 0, 2], &[0, 1, 2, 2]).unwrap();
        assert_abs_diff_eq!(metrics.accuracy, 0.75);
        assert_abs_diff_eq!(metrics.per_class_recall[2].unwrap(), 0.5);
        assert_eq!(metrics.per_class_recall[3], None);
        assert_eq!(metrics.confusion[2][0], 1);
    }
}
