def tightBoundingBoxInkscape(pdffile,use_xvfb=True):
    """Makes POSIX-specific OS calls. Preferably, have xvfb installed, to avoid any GUI popping up in the background. If it fails anyway, could always resort to use_xvfb=False, which will allow some GUIs to show as they carry out the task
      pdffile: the path for a PDF file, without its extension
    """
    usexvfb='xvfb-run '*use_xvfb
    import os
    assert not pdffile.endswith('.pdf')
    os.system("""
        echo %(FN)s
        inkscape -f %(FN)s.pdf -l %(FN)s_tmp.svg
        inkscape -f %(FN)s_tmp.svg --verb=FitCanvasToDrawing \
                                   --verb=FileSave \
                                   --verb=FileQuit
        mkdir -p $(dirname "ready/%(FN)s")
        inkscape -f %(FN)s_tmp.svg -A ready/%(FN)s.pdf
    """ % {'FN':pdffile})

if __name__ == '__main__':
    for query_name in ('range', 'knn', 'knn_bound', 'query_ball'):
        for type_plot in ('blocksize_', 'growing_', 'spatiotemporal_bs_40_', 'spatiotemporal_bs_100_'):
            tightBoundingBoxInkscape(pdffile='out/'+ query_name + '/' + type_plot + query_name)

    for pth in ['knn/knn_vary_k', 'knn_bound/knn_bound_vary_bound', 'query_ball/ball_point_vary_radius']:
        tightBoundingBoxInkscape(pdffile='out/' + pth)