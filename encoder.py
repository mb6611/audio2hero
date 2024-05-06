from typing import List, Optional, Union
from transformers.tokenization_utils import AddedToken, BatchEncoding, PaddingStrategy, PreTrainedTokenizer, TruncationStrategy
import numpy as np
import pretty_midi
def encode_plus(
        tokenizer: PreTrainedTokenizer,
        notes: Union[np.ndarray, List[pretty_midi.Note]],
        truncation_strategy: Optional[TruncationStrategy] = None,
        max_length: Optional[int] = None,
        time_offset: Optional[int] = 0,
        **kwargs,
    ) -> BatchEncoding:
        r"""
        This is the `encode_plus` method for `Pop2PianoTokenizer`. It converts the midi notes to the transformer
        generated token ids. It only works on a single batch, to process multiple batches please use
        `batch_encode_plus` or `__call__` method.

        Args:
            notes (`numpy.ndarray` of shape `[sequence_length, 4]` or `list` of `pretty_midi.Note` objects):
                This represents the midi notes. If `notes` is a `numpy.ndarray`:
                    - Each sequence must have 4 values, they are `onset idx`, `offset idx`, `pitch` and `velocity`.
                If `notes` is a `list` containing `pretty_midi.Note` objects:
                    - Each sequence must have 4 attributes, they are `start`, `end`, `pitch` and `velocity`.
            truncation_strategy ([`~tokenization_utils_base.TruncationStrategy`], *optional*):
                Indicates the truncation strategy that is going to be used during truncation.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).

        Returns:
            `BatchEncoding` containing the tokens ids.
        """

        # check if notes is a pretty_midi object or not, if yes then extract the attributes and put them into a numpy
        # array.

        if len(notes) == 0:
            #  notes = np.array([[0, 1, 55, 77]]).reshape(-1, 4)
            return BatchEncoding({"token_ids": []}), 0

        if isinstance(notes[0], pretty_midi.Note):
            notes = np.array(
                [[each_note.start, each_note.end, each_note.pitch, each_note.velocity] for each_note in notes]
            ).reshape(-1, 4)

        # to round up all the values to the closest int values.
        notes = np.round(notes).astype(np.int32)
        
        max_time_idx = notes[:, :2].max()

        times = [[] for i in range((max_time_idx + 1))]
        for onset, offset, pitch, velocity in notes:
            times[onset].append([pitch, velocity])
            times[offset].append([pitch, 0])

        tokens = []
        current_velocity = 0
        last_time = time_offset
        print(last_time)
        for i, time in enumerate(times):
            if len(time) == 0:
                continue
            
            tokens.append(tokenizer._convert_token_to_id(i - last_time, "TOKEN_TIME"))
            last_time = i
            for pitch, velocity in time:
                velocity = int(velocity > 0)
                if current_velocity != velocity:
                    current_velocity = velocity
                    tokens.append(tokenizer._convert_token_to_id(velocity, "TOKEN_VELOCITY"))
                tokens.append(tokenizer._convert_token_to_id(pitch, "TOKEN_NOTE"))

        total_len = len(tokens)

        # truncation
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            tokens, _, _ = tokenizer.truncate_sequences(
                ids=tokens,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                **kwargs,
            )

        return BatchEncoding({"token_ids": tokens}), last_time - len(times) - 1