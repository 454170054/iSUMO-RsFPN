import tensorflow as tf
from code.feature_extraction.pbf import extract_embedding_features
from code.feature_extraction.zsf import ZScale
from code.feature_extraction.aaindex import AAIndex
import pandas as pd
import numpy as np
from utils import construct_dataset


class Predictor:

    def __init__(self) -> None:
        super().__init__()
        self.model1 = tf.keras.models.load_model("../../weights/model1")
        self.model2 = tf.keras.models.load_model("../../weights/model2")
        self.model3 = tf.keras.models.load_model("../../weights/model3")

    def predict(self, file_path, job_id):
        df = pd.read_csv(file_path, sep=',', header=0)
        AAF = AAIndex(file_path)
        ZSF, label = ZScale(file_path, 1)
        PBF = extract_embedding_features(df['Sequence'].values.tolist())
        p1, p2, p3 = self.model1.predict(AAF)
        p_1 = (p1 + p2 + p3) / 3
        p1, p2, p3 = self.model2.predict(ZSF)
        p_2 = (p1 + p2 + p3) / 3
        p1, p2, p3 = self.model3.predict(PBF)
        p_3 = (p1 + p2 + p3) / 3
        p = (p_1 + p_2 + p_3) / 3
        prediction = np.zeros_like(p)
        prediction[p >= 0.5] = 1
        df['probability'] = p
        df['Label'] = prediction.astype(int)
        print(df)
        df.to_csv(f'../../prediction/results/{job_id}.csv')


if __name__ == '__main__':
    seqs = [
        'MAQKENSYPWPYGRQTAPSGLSTLPQRVLRKEPVTPSALVLMSRSNVQPTAAPGQKVMENSSGTPDILTRHFTIDDFEIGRPLGKGKFGNVYLAREKKSHFIVALKVLFKSQIEKEGVEHQLRREIEIQAHLHHPNILRLYNYFYDRRRIYLILEYAPRGELYKELQKSCTFDEQRTATIMEELADALMYCHGKKVIHRDIKPENLLLGLKGELKIADFGWSVHAPSLRRKTMCGTLDYLPPEMIEGRMHNEKVDLWCIGVLCYELLVGNPPFESASHNETYRRIVKVDLKFPASVPMGAQDLISKLLRHNPSERLPLAQVSAHPWVRANSRRVLPPSALQSVA']
    file_path = construct_dataset(seqs)
    job_id = 'test' # define an id for this job
    predictor = Predictor()
    predictor.predict(file_path, job_id)
