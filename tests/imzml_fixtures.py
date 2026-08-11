"""Hand-rolled, minimal imzML + .ibd writer for tests.

Deliberately does not use pyimzml.ImzMLWriter (which depends on the extra
`wheezy.template` package and would just be testing pyimzml's own writer
against its own reader). This writes the XML and binary layout directly, by
hand, against the imzML 1.1.0 spec, so the tests exercise this project's
reader code (msi_io.read_binary_array, pass1/pass2/common's use of
mzOffsets/mzLengths/mzPrecision) against byte-for-byte known input.

Only the elements pyimzml.ImzMLParser actually requires are included -
fileDescription/referenceableParamGroupList/scanSettingsList/
instrumentConfigurationList/dataProcessingList/run/spectrumList - each
spectrum declaring its m/z + intensity array precision, offset, and element
count (not byte count - pyimzml multiplies by itemsize itself).
"""
from pathlib import Path

import numpy as np

NS = "http://psi.hupo.org/ms/mzml"
PRECISION_ACCESSION = {
    "f": ("MS:1000521", "32-bit float"),
    "d": ("MS:1000523", "64-bit float"),
}
SPECTRUM_TYPE_ACCESSION = {
    "profile": ("MS:1000128", "profile spectrum"),
    "centroid": ("MS:1000127", "centroid spectrum"),
}


def _spectrum_xml(index, x, y, mz_off, mz_len, mz_enc_len, int_off, int_len, int_enc_len, spectrum_type=None):
    type_cvparam = ""
    if spectrum_type is not None:
        acc, name = SPECTRUM_TYPE_ACCESSION[spectrum_type]
        type_cvparam = f'\n        <cvParam cvRef="MS" accession="{acc}" name="{name}" value=""/>'
    return f"""      <spectrum defaultArrayLength="0" id="spectrum={index}" index="{index}">
        <cvParam cvRef="MS" accession="MS:1000579" name="MS1 spectrum" value=""/>{type_cvparam}
        <scanList count="1">
          <cvParam cvRef="MS" accession="MS:1000795" name="no combination"/>
          <scan instrumentConfigurationRef="IC1">
            <cvParam cvRef="IMS" accession="IMS:1000050" name="position x" value="{x}"/>
            <cvParam cvRef="IMS" accession="IMS:1000051" name="position y" value="{y}"/>
          </scan>
        </scanList>
        <binaryDataArrayList count="2">
          <binaryDataArray encodedLength="0">
            <referenceableParamGroupRef ref="mzArray"/>
            <cvParam cvRef="IMS" accession="IMS:1000103" name="external array length" value="{mz_len}"/>
            <cvParam cvRef="IMS" accession="IMS:1000104" name="external encoded length" value="{mz_enc_len}"/>
            <cvParam cvRef="IMS" accession="IMS:1000102" name="external offset" value="{mz_off}"/>
            <binary/>
          </binaryDataArray>
          <binaryDataArray encodedLength="0">
            <referenceableParamGroupRef ref="intensityArray"/>
            <cvParam cvRef="IMS" accession="IMS:1000103" name="external array length" value="{int_len}"/>
            <cvParam cvRef="IMS" accession="IMS:1000104" name="external encoded length" value="{int_enc_len}"/>
            <cvParam cvRef="IMS" accession="IMS:1000102" name="external offset" value="{int_off}"/>
            <binary/>
          </binaryDataArray>
        </binaryDataArrayList>
      </spectrum>"""


def _document(mz_precision, intensity_precision, maxx, maxy, n_spectra, spectra_xml):
    mz_acc, mz_name = PRECISION_ACCESSION[mz_precision]
    int_acc, int_name = PRECISION_ACCESSION[intensity_precision]
    return f"""<?xml version="1.0" encoding="ISO-8859-1"?>
<mzML xmlns="{NS}" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="{NS} http://psidev.info/files/ms/mzML/xsd/mzML1.1.0_idx.xsd" version="1.1">
  <fileDescription>
    <fileContent>
      <cvParam cvRef="MS" accession="MS:1000579" name="MS1 spectrum" value=""/>
    </fileContent>
  </fileDescription>
  <referenceableParamGroupList count="2">
    <referenceableParamGroup id="mzArray">
      <cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>
      <cvParam cvRef="MS" accession="MS:1000514" name="m/z array" unitCvRef="MS" unitAccession="MS:1000040" unitName="m/z"/>
      <cvParam cvRef="MS" accession="{mz_acc}" name="{mz_name}" value=""/>
      <cvParam cvRef="IMS" accession="IMS:1000101" name="external data" value="true"/>
    </referenceableParamGroup>
    <referenceableParamGroup id="intensityArray">
      <cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>
      <cvParam cvRef="MS" accession="MS:1000515" name="intensity array" unitCvRef="MS" unitAccession="MS:1000131" unitName="number of detector counts"/>
      <cvParam cvRef="MS" accession="{int_acc}" name="{int_name}" value=""/>
      <cvParam cvRef="IMS" accession="IMS:1000101" name="external data" value="true"/>
    </referenceableParamGroup>
  </referenceableParamGroupList>
  <softwareList count="1">
    <software id="test_writer" version="1.0">
      <cvParam cvRef="MS" accession="MS:1000799" name="custom unreleased software tool" value="test fixture writer"/>
    </software>
  </softwareList>
  <scanSettingsList count="1">
    <scanSettings id="scanSettings1">
      <cvParam cvRef="IMS" accession="IMS:1000042" name="max count of pixels x" value="{maxx}"/>
      <cvParam cvRef="IMS" accession="IMS:1000043" name="max count of pixels y" value="{maxy}"/>
    </scanSettings>
  </scanSettingsList>
  <instrumentConfigurationList count="1">
    <instrumentConfiguration id="IC1"></instrumentConfiguration>
  </instrumentConfigurationList>
  <dataProcessingList count="1">
    <dataProcessing id="dp1">
      <processingMethod order="0" softwareRef="test_writer">
        <cvParam cvRef="MS" accession="MS:1000530" name="file format conversion" value="Output to imzML"/>
      </processingMethod>
    </dataProcessing>
  </dataProcessingList>
  <run defaultInstrumentConfigurationRef="IC1" id="run1">
    <spectrumList count="{n_spectra}" defaultDataProcessingRef="dp1">
{spectra_xml}
    </spectrumList>
  </run>
</mzML>
"""


def write_mode_folder(mode_dir, pixels, mz_precision="f", intensity_precision="f", stem="sample", spectrum_type=None):
    """pixels: list of (x, y, mz_array, intensity_array). Writes <mode_dir>/<stem>.imzML + .ibd.
    spectrum_type: "profile" / "centroid" / None to declare (or omit) the
    per-spectrum MS:1000128/MS:1000127 cvParam that ImzMLParser.spectrum_mode
    reads - needed to test flat-layout mode auto-detection.
    Returns (imzml_path, ibd_path)."""
    mode_dir = Path(mode_dir)
    mode_dir.mkdir(parents=True, exist_ok=True)
    imzml_path = mode_dir / f"{stem}.imzML"
    ibd_path = mode_dir / f"{stem}.ibd"
    mz_dtype = np.dtype(mz_precision)
    int_dtype = np.dtype(intensity_precision)

    spectra_blocks = []
    offset = 0
    with open(ibd_path, "wb") as fh:
        for idx, (x, y, mz, it) in enumerate(pixels, start=1):
            mz_bytes = np.asarray(mz, dtype=mz_dtype).tobytes()
            it_bytes = np.asarray(it, dtype=int_dtype).tobytes()
            mz_off = offset
            fh.write(mz_bytes)
            offset += len(mz_bytes)
            int_off = offset
            fh.write(it_bytes)
            offset += len(it_bytes)
            spectra_blocks.append(_spectrum_xml(
                idx, x, y, mz_off, len(mz), len(mz_bytes), int_off, len(it), len(it_bytes), spectrum_type=spectrum_type))

    maxx = max(p[0] for p in pixels)
    maxy = max(p[1] for p in pixels)
    xml = _document(mz_precision, intensity_precision, maxx, maxy, len(pixels), "\n".join(spectra_blocks))
    imzml_path.write_text(xml, encoding="ISO-8859-1")
    return imzml_path, ibd_path


def gaussian_profile_pixel(targets, halfwin, grid, width=0.0008, amp=1000.0):
    """One profile spectrum: a small Gaussian peak around each target, concatenated and sorted."""
    mzs, its = [], []
    for T in targets:
        g = np.arange(T - halfwin, T + halfwin, grid)
        y = amp * np.exp(-0.5 * ((g - T) / width) ** 2)
        mzs.append(g)
        its.append(y)
    mz = np.concatenate(mzs)
    it = np.concatenate(its)
    order = np.argsort(mz)
    return mz[order], it[order]


def centroid_pixel(targets, amp=800.0):
    mz = np.array(sorted(targets), dtype=np.float64)
    it = np.full(len(targets), amp, dtype=np.float64)
    return mz, it
