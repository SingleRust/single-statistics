pub fn getset(cnct: &[usize], starts: &[usize], offsets: &[usize], j: usize) -> Vec<usize> {
    let srt = starts[j];
    let off = srt + offsets[j];
    let fset = &cnct[srt..off];
    fset.to_vec()
}

