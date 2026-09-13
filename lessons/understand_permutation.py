"""A complete ten-component teaching example, then a 4,096-dimensional check."""
import torch
import torchhd
from graph_to_hypervector.encoding import Encoder, bundle


def main():
    # This is a COMPLETE toy hypervector, not the first ten of a longer one.
    h_example = torch.tensor([1., -1., -1., 1., -1., 1., 1., -1., 1., -1.]).as_subclass(torchhd.MAPTensor)
    h_once = torchhd.permute(h_example, shifts=1)  # Right by exactly one component.
    h_twice = torchhd.permute(h_once, shifts=1)    # The same single offset again.
    print("Original: ", h_example.tolist())
    print("One shift:", h_once.tolist())
    print("Two shifts:", h_twice.tolist())
    assert torch.equal(h_twice, torchhd.permute(h_example, shifts=2))

    # REVERSAL: left by one, twice, exactly reverses the two right shifts.
    h_restored = torchhd.permute(torchhd.permute(h_twice, shifts=-1), shifts=-1)
    assert torch.equal(h_restored, h_example)
    print("Restored: ", h_restored.tolist())

    # The actual path uses the full 4,096 components as its wraparound boundary.
    encoder = Encoder()
    h_nina = encoder.properties({"age": 33, "eye_color": "blue",
                                 "interests": ["tea", "cooking", "photography"]})
    h_positioned = torchhd.permute(h_nina, shifts=2)
    assert torch.equal(torchhd.permute(h_positioned, shifts=-2), h_nina)

    # Reversing permutation is not the same as extracting from a bundle.
    h_maya = encoder.properties({"age": 34, "eye_color": "brown",
                                 "interests": ["tea", "climbing", "jazz"]})
    h_bundle = bundle([h_maya, h_positioned])
    h_unshifted_bundle = torchhd.permute(h_bundle, shifts=-2)
    assert torch.equal(h_unshifted_bundle, bundle([torchhd.permute(h_maya, shifts=-2), h_nina]))
    print("Undoing a bundle's shift moves ALL its contributions; Maya's remains present.")


if __name__ == "__main__":
    main()
