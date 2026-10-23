import io
import zipfile
import numpy as np
import pytest
from PIL import Image
from app.io import load_volume,DataError
from app.analysis import vessel_and_faz


def test_zip_numeric_sort(tmp_path):
    file=tmp_path/'stack.zip'
    with zipfile.ZipFile(file,'w') as z:
        for index in (10,2,1):
            buf=io.BytesIO();Image.fromarray(np.full((64,64),index,dtype=np.uint8)).save(buf,format='PNG')
            z.writestr(f'image.{index:04}.png',buf.getvalue())
    v=load_volume(file)
    assert v.shape==(3,64,64)
    assert v[:,0,0].tolist()==[1,2,10]


def test_zip_invalid_shape(tmp_path):
    file=tmp_path/'stack.zip'
    with zipfile.ZipFile(file,'w') as z:
        for i,shape in enumerate([(64,64),(64,65)]):
            buf=io.BytesIO();Image.fromarray(np.zeros(shape,dtype=np.uint8)).save(buf,format='PNG');z.writestr(f'{i}.png',buf.getvalue())
    with pytest.raises(DataError,match='mismatch'):load_volume(file)


def test_faz_with_enclosing_vessels_and_calibration():
    import cv2
    a=np.zeros((160,160),np.uint8)
    cv2.circle(a,(80,80),28,250,3)
    # dense surrounding signal, creating single large enclosed gap
    m,_,_=vessel_and_faz(a,52,80,80,3.0,3.0)
    assert m['faz_detected']
    assert m['faz_area_px']>0
    assert m['faz_area_mm2'] is not None


def test_faz_no_false_area_for_empty_image():
    m,_,_=vessel_and_faz(np.zeros((100,100),np.uint8))
    assert m['faz_detected'] is False
    assert m['faz_area_px'] is None
    assert m['faz_area_mm2'] is None


def test_no_calibration_means_no_physical_units():
    m,_,_=vessel_and_faz(np.zeros((100,100),np.uint8))
    assert m['faz_area_mm2'] is None
    assert m['vessel_length_mm_proxy'] is None
