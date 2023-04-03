'''
This file was created by ]init[ AG 2022.

Tests for Whatever.
'''
import logging
from qa_service.t2t_manager.manager import T2tManager

log = logging.getLogger(__name__)


def test_question():
    t2tManager = T2tManager()
    log.debug(t2tManager.generate('''Frage: Gibt es besondere Regelungen für Elterngeld bei Frühgeborenen?
Antworte in Deutsch mit maximal 3 Sätzen und nutze dabei ausschließlich folgende Informationen:
-----
Welche Besonderheiten gelten bei Frühchen?
Wenn Ihr Kind mindestens 6 Wochen vor dem errechneten Geburtstermin zur Welt kommt, können Sie länger Elterngeld bekommen. Dieser längere Bezug ist nur möglich, wenn das Kind ab dem 1. September 2021 geboren ist. Bis zu 4 zusätzliche Monate Basiselterngeld sind möglich, abhängig vom Geburtstermin: bei einer Geburt mindestens 6 Wochen vor dem errechneten Termin: 1 zusätzlicher Monat Basiselterngeld bei einer Geburt mindestens 8 Wochen vor dem errechneten Termin: 2 zusätzliche Monate Basiselterngeld bei einer Geburt mindestens 12 Wochen vor dem errechneten Termin: 3 zusätzliche Monate Basiselterngeld bei einer Geburt mindestens 16 Wochen vor dem errechneten Termin: 4 zusätzliche Monate BasiselterngeldWie sonst auch können Sie jeden dieser zusätzlichen Monate mit Basiselterngeld tauschen gegen jeweils 2 Monate mit ElterngeldPlus.Für diese zusätzlichen Monate werden auch Ihre Gestaltungs-Möglichkeiten erweitert: Bei einem zusätzlichen Monat können Sie Basiselterngeld in den ersten 15 Lebensmonaten bekommen. Erst ab dem 16. Lebensmonat darf der Elterngeld-Bezug nicht mehr unterbrochen werden. Bei 2 zusätzlichen Monaten können Sie Basiselterngeld in den ersten 16 Lebensmonaten bekommen. Erst ab dem 17. Lebensmonat darf der Elterngeld-Bezug nicht mehr unterbrochen werden. Bei 3 zusätzlichen Monaten können Sie Basiselterngeld in den ersten 17 Lebensmonaten bekommen. Erst ab dem 18. Lebensmonat darf der Elterngeld-Bezug nicht mehr unterbrochen werden. Bei 4 zusätzlichen Monaten können Sie Basiselterngeld in den ersten 18 Lebensmonaten bekommen. Erst ab dem 19. Lebensmonat darf der Elterngeld-Bezug nicht mehr unterbrochen werden.
-----
Antwort:'''))
